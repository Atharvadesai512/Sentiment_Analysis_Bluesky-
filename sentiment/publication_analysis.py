"""Secondary quantification diagnostic and historical robustness audit.

No model fitting, new labels, or target-domain accuracy estimates are introduced.
"""
from pathlib import Path
from itertools import combinations
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/publication/evidence'
MODELS = ['vader', 'linear', 'bilstm_saved', 'roberta']
SCORE = np.array([-1., 0., 1.])


def constrained_acc(matrix, observed):
    """Minimize ||A p-q||^2 on the 3-class simplex by enumerating all faces."""
    observed = np.atleast_2d(observed)
    best = np.full(len(observed), np.inf)
    result = np.zeros_like(observed)
    for size in [1, 2, 3]:
        for active in combinations(range(3), size):
            sub = matrix[:, active]
            kkt = np.block([[sub.T @ sub, np.ones((size, 1))],
                            [np.ones((1, size)), np.zeros((1, 1))]])
            rhs = np.vstack([sub.T @ observed.T, np.ones((1, len(observed)))])
            weights = np.linalg.lstsq(kkt, rhs, rcond=None)[0][:-1].T
            valid = (weights >= -1e-10).all(axis=1)
            candidate = np.zeros_like(observed)
            candidate[:, active] = np.maximum(weights, 0)
            candidate /= candidate.sum(axis=1, keepdims=True)
            loss = ((candidate @ matrix.T - observed)**2).sum(axis=1)
            update = valid & (loss < best)
            result[update] = candidate[update]
            best[update] = loss[update]
    assert np.isfinite(best).all()
    assert np.allclose(result.sum(axis=1), 1) and (result >= 0).all()
    return result


def quantification():
    frames = {m: pd.read_csv(ROOT / f'research/benchmark/{m}_predictions.csv') for m in MODELS}
    reference = frames[MODELS[0]]
    for frame in frames.values():
        assert frame[['row_id', 'label']].equals(reference[['row_id', 'label']])
    gold = reference.label.to_numpy()
    predictions = {m: f.prediction.to_numpy() for m, f in frames.items()}
    # Deterministic stratification, with every original test row in exactly one half.
    rng = np.random.default_rng(42)
    halves = [[], []]
    for label in range(3):
        ids = rng.permutation(np.flatnonzero(gold == label))
        cut = len(ids) // 2
        halves[0].extend(ids[:cut]); halves[1].extend(ids[cut:])
    halves = [np.array(sorted(h)) for h in halves]
    assert not set(halves[0]) & set(halves[1])
    pd.DataFrame({'row_id': reference.row_id, 'half': np.isin(np.arange(len(gold)), halves[1]).astype(int)}).to_csv(OUT/'diagnostic_split.csv', index=False)
    grid = np.array([[a, b, 10-a-b] for a in range(11) for b in range(11-a)]) / 10
    matrices, results = {}, []
    for direction, calibration, test in [(0, halves[0], halves[1]), (1, halves[1], halves[0])]:
        for model in MODELS:
            matrix = np.array([[np.mean(predictions[model][calibration][gold[calibration] == j] == i)
                                for j in range(3)] for i in range(3)])
            matrices[f'{direction}/{model}'] = {'matrix_predicted_by_true': matrix.tolist(),
                                              'condition_number': float(np.linalg.cond(matrix))}
        pools = [test[gold[test] == j] for j in range(3)]
        for size in ([100, 500, 1000] if direction == 0 else [500]):
            bag_rng = np.random.default_rng(1000 + direction*10000 + size)
            bags = []
            for grid_id, prevalence in enumerate(grid):
                for repeat in range(30):
                    ids = np.concatenate([bag_rng.choice(pools[j], int(round(size*prevalence[j])), replace=True) for j in range(3)])
                    assert len(ids) == size
                    bags.append((grid_id, repeat, prevalence, ids))
            for model in MODELS:
                observed = np.array([np.bincount(predictions[model][ids], minlength=3)/size for _, _, _, ids in bags])
                estimates = {'CC': observed, 'ACC': constrained_acc(np.array(matrices[f'{direction}/{model}']['matrix_predicted_by_true']), observed)}
                for method, values in estimates.items():
                    for (grid_id, repeat, prevalence, _), estimated in zip(bags, values):
                        target = float(prevalence @ SCORE)
                        net = float(estimated @ SCORE)
                        results.append({'direction': direction, 'bag_size': size, 'grid_id': grid_id,
                                        'repeat': repeat, 'model': model, 'method': method,
                                        'true_negative': prevalence[0], 'true_neutral': prevalence[1], 'true_positive': prevalence[2],
                                        'pred_negative': estimated[0], 'pred_neutral': estimated[1], 'pred_positive': estimated[2],
                                        'prevalence_mae': float(np.abs(prevalence-estimated).mean()),
                                        'net_absolute_error': abs(net-target), 'net_signed_error': net-target,
                                        'true_net': target, 'pred_net': net,
                                        'strict_sign_error': int(net*target < -1e-12),
                                        'eligible_for_sign': int(abs(target) >= .1-1e-12)})
    data = pd.DataFrame(results)
    data.to_csv(OUT/'quantification_bags.csv', index=False)
    aggregates = []
    for key, group in data.groupby(['direction', 'bag_size', 'model', 'method']):
        sign = group[group.eligible_for_sign == 1]
        aggregates.append(dict(zip(['direction', 'bag_size', 'model', 'method'], key)) | {
            'bags': len(group), 'prevalence_mae': float(group.prevalence_mae.mean()),
            'net_mae': float(group.net_absolute_error.mean()), 'net_bias': float(group.net_signed_error.mean()),
            'strict_sign_error_rate': float(sign.strict_sign_error.mean()), 'sign_eligible_bags': len(sign)})
    pd.DataFrame(aggregates).to_csv(OUT/'quantification_summary.csv', index=False)
    (OUT/'confusion_calibration.json').write_text(json.dumps(matrices, indent=2))
    return {'half_sizes': list(map(len, halves)), 'prevalence_grid_points': len(grid), 'repeats':30,
            'primary_bag_size':500, 'primary_direction':0, 'seed_split':42,
            'sampling':'With replacement within true class; identical bags for every method; grid equally weighted',
            'calibration':'Half of the original test rows calibrate confusion; the disjoint half supplies bags. Reverse-direction sensitivity at n=500. Base classifiers unchanged.',
            'scope':'Retrospective secondary diagnostic on a previously inspected test set; not a new independent benchmark or target-domain correction.',
            'matrices_orientation':'rows predicted class, columns true class; order negative, neutral, positive'}


def robustness():
    from research_results import load_predictions
    corpus = pd.read_csv(ROOT/'data/cleaned_posts.csv').set_index('uri', verify_integrity=True)
    labels = load_predictions(ROOT/'data', corpus)
    scores = labels.apply(lambda col: col.map({'negative':-1, 'neutral':0, 'positive':1})).astype(float)
    dates = pd.to_datetime(corpus.created_at, format='ISO8601', utc=True)
    author = corpus.index.to_series().str.extract(r'^at://([^/]+)/', expand=False)
    assert author.notna().all()
    day = dates.dt.floor('D'); hour = dates.dt.floor('h')
    normalized = corpus.text.str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    order = pd.DataFrame({'time': dates, 'post_uri':corpus.index}, index=corpus.index).sort_values(['time','post_uri'])
    keep = normalized.loc[order.index].drop_duplicates().index
    coverage = pd.read_csv(ROOT/'research/temporal/sentiment_h.csv')
    good = coverage[(coverage.model=='VADER') & coverage.fully_observed_uncapped.astype(bool)].period_start_utc
    good = pd.to_datetime(good, utc=True)
    masks = {'all_posts':pd.Series(True,index=corpus.index),
             'unique_normalized_text':pd.Series(corpus.index.isin(keep),index=corpus.index),
             'uncapped_hours':hour.isin(good),
             'literal_ai_token':corpus.text.str.contains(r'\bAI\b',case=False,regex=True)}
    rows=[]
    for name, mask in masks.items():
        for model in scores:
            rows.append({'analysis':name,'model':model,'units':int(mask.sum()),'net_sentiment':float(scores.loc[mask,model].mean())})
    author_day = scores.assign(author=author,day=day).groupby(['day','author']).mean()
    author_all = scores.assign(author=author).groupby('author').mean()
    for name, frame in [('equal_author_day',author_day),('equal_author',author_all)]:
        for model in scores:
            rows.append({'analysis':name,'model':model,'units':len(frame),'net_sentiment':float(frame[model].mean())})
    pd.DataFrame(rows).to_csv(OUT/'historical_robustness.csv',index=False)
    # Exact accounting identity for mean label-score disagreement, not a novel estimator.
    contributions=[]
    for left,right in combinations(scores.columns,2):
        for a in ['negative','neutral','positive']:
            for b in ['negative','neutral','positive']:
                count=int(((labels[left]==a)&(labels[right]==b)).sum())
                delta={'negative':-1,'neutral':0,'positive':1}[a]-{'negative':-1,'neutral':0,'positive':1}[b]
                contributions.append({'left':left,'right':right,'left_label':a,'right_label':b,'posts':count,
                                      'contribution_to_mean_difference':count*delta/len(corpus)})
    pd.DataFrame(contributions).to_csv(OUT/'disagreement_decomposition.csv',index=False)
    daily=scores.groupby(day).mean()
    daily.to_csv(OUT/'daily_mean_scores.csv')
    differences = daily.VADER-daily.RoBERTa
    rng=np.random.default_rng(42)
    boot=np.mean(rng.choice(differences.to_numpy(),size=(10000,len(differences)),replace=True),axis=1)
    return {'posts':len(corpus),'authors':int(author.nunique()),'unique_normalized_texts':len(keep),
            'author_days':len(author_day),'top_one_percent_authors_post_share':float(author.value_counts().head(int(np.ceil(author.nunique()*.01))).sum()/len(corpus)),
            'daily_equal_weight_vader_minus_roberta':float(differences.mean()),
            'day_resampling_95_interval':np.quantile(boot,[.025,.975]).tolist(),
            'daily_positive_difference_count':int((differences>0).sum()),'days':len(daily),
            'interval_scope':'10000 resamples of 14 observed day means; seed42; assumes exchangeable days; descriptive, not population inference or classification-correctness uncertainty.'}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # Verify the optimizer against simple exact identities before using it.
    q=np.array([[.2,.3,.5],[0,1,0],[1,0,0]])
    assert np.allclose(constrained_acc(np.eye(3),q),q)
    summary={'quantification_protocol':quantification(),'historical':robustness()}
    summary['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'analysis_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
