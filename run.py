"""Reproducible synthetic active evidence acquisition; no enterprise data."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

COST = np.tile([1., 2., 3.], 4)
NOISE = np.tile([1.2, .7, .4], 4)
GROUP = np.repeat(np.arange(4), 3)
POLICIES = ['Random', 'Cheap first', 'Fixed gain', 'No knowledge', 'No cost', 'KGEA']

def design(z):
    return np.column_stack([z, np.min(z[:, [0, 1, 3]], axis=1), z[:, 2]*z[:, 3]])

def sample(rng, n, scenario):
    z = rng.normal(size=(n, 4))
    if scenario == 'Shift':
        z = z * np.array([1.5, .6, 1.2, .8]) + [.4, -.5, .3, -.2]
    x = z[:, GROUP] + rng.normal(size=(n, 12))*NOISE
    nonlinear = 1.4*np.min(z[:, [0, 1, 3]], axis=1) + .8*z[:, 2]*z[:, 3]
    if scenario == 'Reversed rules':
        nonlinear *= -1
    p = expit(.2 + z @ np.array([.8, .7, .4, .9]) + nonlinear)
    return x, rng.binomial(1, p)

def posterior(x, mask):
    # np.where prevents hidden NaNs or changed hidden values influencing estimates.
    precision = mask / NOISE**2
    v = 1/(1 + precision.reshape(-1, 4, 3).sum(axis=2))
    mu = (np.where(mask, x, 0)/NOISE**2).reshape(-1, 4, 3).sum(axis=2)*v
    return mu, v

def scores(mu, v, coef, policy):
    gradient = np.broadcast_to(coef[:4], mu.shape).copy()
    if policy != 'No knowledge':
        bottleneck = np.array([0, 1, 3])[np.argmin(mu[:, [0, 1, 3]], axis=1)]
        gradient[np.arange(len(mu)), bottleneck] += coef[4]
        gradient[:, 2] += coef[5]*mu[:, 3]
        gradient[:, 3] += coef[5]*mu[:, 2]
    gain = gradient[:, GROUP]**2 * v[:, GROUP]**2/(v[:, GROUP]+NOISE**2)
    return gain if policy == 'No cost' else gain/COST

def acquire(x, model, budget, policy, rng):
    mask = np.zeros(x.shape, bool)
    spent = np.zeros(len(x))
    fixed = scores(np.zeros((1, 4)), np.ones((1, 4)), model.coef_[0], 'KGEA')
    for _ in range(12):
        mu, v = posterior(x, mask)
        feasible = (~mask) & (COST[None, :] <= (budget-spent)[:, None]+1e-10)
        active = feasible.any(axis=1)
        if not active.any():
            break
        if policy == 'Random':
            s = rng.random(x.shape)
        elif policy == 'Cheap first':
            s = np.broadcast_to(-COST, x.shape)
        elif policy == 'Fixed gain':
            s = np.broadcast_to(fixed, x.shape)
        else:
            s = scores(mu, v, model.coef_[0], policy)
        j = np.argmax(np.where(feasible, s, -np.inf), axis=1)
        rows = np.flatnonzero(active)
        mask[rows, j[rows]] = True
        spent[rows] += COST[j[rows]]
    assert np.all(spent <= budget+1e-10)
    mu, _ = posterior(x, mask)
    return model.predict_proba(design(mu))[:, 1], spent

def checks():
    rng = np.random.default_rng(71)
    x, y = sample(rng, 100, 'Matched')
    mask = rng.random(x.shape) < .5
    altered = np.where(mask, x, np.nan)
    a, b = posterior(x, mask), posterior(altered, mask)
    for u, v in zip(a, b):
        np.testing.assert_allclose(u, v)
    mu, v = posterior(x, np.zeros_like(mask))
    assert np.all(mu == 0) and np.all(v == 1)
    model = LogisticRegression().fit(design(posterior(x, np.ones_like(mask))[0]), y)
    for policy in POLICIES:
        p, spent = acquire(x, model, 0, policy, rng)
        assert np.all(spent == 0) and np.ptp(p) == 0
    print('Visibility, empty-posterior and zero-budget checks passed.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--out', type=Path, default=Path('results'))
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error('--seeds must be at least 2')
    checks()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(1000+seed)
        train, ytrain = sample(rng, 2000, 'Matched')
        mu, _ = posterior(train, np.ones_like(train, bool))
        model = LogisticRegression(C=1., max_iter=1000).fit(design(mu), ytrain)
        for scenario in ['Matched', 'Reversed rules', 'Shift']:
            x, y = sample(rng, 500, scenario)
            for policy in POLICIES:
                for budget in range(10):
                    pred, cost = acquire(x, model, budget, policy, np.random.default_rng(seed*100+budget))
                    records.append([seed, scenario, policy, budget, brier_score_loss(y, pred), cost.mean()])
    df = pd.DataFrame(records, columns=['seed', 'scenario', 'policy', 'budget', 'brier', 'cost'])
    df.to_csv(args.out/'results.csv', index=False)
    integrated = df.groupby(['seed', 'scenario', 'policy']).apply(lambda g: np.trapz(g.sort_values('budget').brier, g.sort_values('budget').budget)/9).rename('iBrier').reset_index()
    summary = integrated.groupby(['scenario', 'policy']).iBrier.agg(['mean', 'std']).reset_index()
    summary.to_csv(args.out/'table1.csv', index=False)
    paired = integrated.pivot(index=['seed', 'scenario'], columns='policy', values='iBrier')
    contrasts = []
    for scenario in ['Matched', 'Reversed rules', 'Shift']:
        for policy in POLICIES[:-1]:
            delta = paired.xs(scenario, level='scenario')[policy]-paired.xs(scenario, level='scenario')['KGEA']
            from scipy.stats import t
            half = t.ppf(.975, len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))
            contrasts.append([scenario, policy, delta.mean(), delta.mean()-half, delta.mean()+half])
    contrasts = pd.DataFrame(contrasts, columns=['scenario', 'comparator', 'improvement', 'lower', 'upper'])
    contrasts.to_csv(args.out/'table2.csv', index=False)
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, scenario in zip(axes, ['Matched', 'Reversed rules', 'Shift']):
        for policy in POLICIES:
            g = df[(df.scenario == scenario)&(df.policy == policy)].groupby('budget').brier.mean()
            ax.plot(g.index, g, label=policy, linewidth=2 if policy == 'KGEA' else 1)
        ax.set(title=scenario, xlabel='Evidence budget'); ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean held-out Brier score')
    axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.out/'Figure_1.svg'); fig.savefig(args.out/'Figure_1.png', dpi=300); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, scenario in zip(axes, ['Matched', 'Reversed rules', 'Shift']):
        g = contrasts[contrasts.scenario == scenario]
        ax.errorbar(g.improvement, np.arange(5), xerr=[g.improvement-g.lower, g.upper-g.improvement], fmt='o', capsize=3)
        ax.axvline(0, color='gray', linestyle='--'); ax.set_yticks(np.arange(5), g.comparator)
        ax.set(title=scenario, xlabel='Comparator iBrier minus KGEA'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(args.out/'Figure_2.svg'); fig.savefig(args.out/'Figure_2.png', dpi=300)
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
