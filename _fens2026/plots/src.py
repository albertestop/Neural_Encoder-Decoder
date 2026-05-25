import numpy as np
import pickle
import importlib
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.linalg import null_space
import os
import pandas as pd


ALPHA_VALUE = 0.6
SIZE = 4

def get_pupil_categories_t(proc_config):
    with open(os.path.join(proc_config.data['session_dir'], 'recordings', 'dlcEyeLeft.pickle'), "rb") as file:
        left_eyedat = pickle.load(file)
    with open(os.path.join(proc_config.data['session_dir'], 'recordings', 'dlcEyeRight.pickle'), "rb") as file:
        right_eyedat = pickle.load(file)
    eye_t = np.load(os.path.join(proc_config.data['session_dir'], 'recordings/eye_frame_times.npy'))


    left_r = np.asarray(left_eyedat.get("radius", []), dtype=float)
    n = min(eye_t.size, left_r.size)
    if n > 0:
        left_d = 2.0 * left_r[:n]
        finite_left = left_d[np.isfinite(left_d)]
        if finite_left.size > 0:
            left_cap = float(np.percentile(finite_left, 80.0))
            left_d = np.minimum(left_d, left_cap)

    right_r = np.asarray(right_eyedat.get("radius", []), dtype=float)
    n = min(eye_t.size, right_r.size)
    if n > 0:
        right_d = 2.0 * right_r[:n]
        finite_right = right_d[np.isfinite(right_d)]
        if finite_right.size > 0:
            right_cap = float(np.percentile(finite_right, 80.0))
            right_d = np.minimum(right_d, right_cap)

    if np.isnan(right_d).sum() < np.isnan(left_d).sum():
        pupil_dilation = right_d
    else:
        pupil_dilation = left_d

    nans = np.isnan(pupil_dilation)
    x = np.arange(len(pupil_dilation))
    pupil_dilation[nans] = np.interp(x[nans], x[~nans], pupil_dilation[~nans])
    N = len(pupil_dilation)
    order = np.sort(pupil_dilation)

    idx_10 = int(0.1 * N)
    idx_30 = int(0.3 * N)
    idx_70 = int(0.7 * N)

    mask_70_100 = pupil_dilation >= order[idx_70]
    mask_30_70 = (pupil_dilation >= order[idx_30]) & (pupil_dilation < order[idx_70])
    mask_10_30 = (pupil_dilation >= order[idx_10]) & (pupil_dilation < order[idx_30])
    mask_0_10 = pupil_dilation < order[idx_10]

    return [eye_t[mask_0_10], eye_t[mask_10_30], eye_t[mask_30_70], eye_t[mask_70_100]]

def get_movie_categories_t(trials_df, skipped_indexes):
    seen_trials_df = trials_df.drop(index=skipped_indexes).reset_index(drop=True)
    not_seen_trials_df = trials_df.loc[skipped_indexes].reset_index(drop=True)

    seen_trials_t = []
    inter_trial_t = []
    not_seen_trials_t = []
    for time, duration in zip(seen_trials_df['time'], seen_trials_df['duration']):
        seen_trials_t.append([time, time + duration])
        inter_trial_t.append([time + duration, time + duration + 5])
    for time, duration in zip(not_seen_trials_df['time'], not_seen_trials_df['duration']):
        not_seen_trials_t.append([time, time + duration])
        inter_trial_t.append([time + duration, time + duration + 5])

    seen_trials_t = np.array(seen_trials_t)
    inter_trial_t = np.array(inter_trial_t)
    not_seen_trials_t = np.array(not_seen_trials_t)
    
    return seen_trials_t, inter_trial_t, not_seen_trials_t


def get_sleep_categories_t(session):
    
    with open(session / Path("sleep_score/sleep_state.pickle"), "rb") as f:
        data = pickle.load(f)
    state_labels = data["state_labels"]
    sleep_state_t = data["state_10hz_t"]
    sleep_state = data["state_10hz"]
    active_wake_t = sleep_state_t[sleep_state == 0]
    quiet_wake_t = sleep_state_t[sleep_state == 1]
    nrem_t = sleep_state_t[sleep_state == 2]
    rem_t = sleep_state_t[sleep_state == 3]

    return [active_wake_t, quiet_wake_t, nrem_t, rem_t]


def load_config(path):
    spec = importlib.util.spec_from_file_location("rec_config", path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def segment_movie_session(categories_t, timeline, metrics_raw):
    """
        Returns a list organised as total_metrics[category][metric]
    """
    total_metrics = []
    total_metrics_t = []
    for category in categories_t:
        cat_metrics = []
        cat_metrics_t = np.empty(0)
        for i in range(len(metrics_raw)):
            cat_metrics.append(np.empty(0))

        for t_i, t_f in category:
            mask = (timeline >= t_i) & (timeline <= t_f)
            cat_metrics_t = np.concatenate((cat_metrics_t, timeline[mask]))
            for i in range(len(metrics_raw)):
                cat_metrics[i] = np.concatenate((cat_metrics[i], metrics_raw[i][mask, 1]))
                

        total_metrics.append(cat_metrics)
        total_metrics_t.append(cat_metrics_t)

    return total_metrics, total_metrics_t


def segment_sleep_session(categories_t, timeline, metrics_raw):
    """
        Returns a list organised as total_metrics[category][metric]
    """
    total_metrics = []
    total_metrics_t = []
    for category_t in categories_t:
        cat_metrics = []
        cat_metrics_t = np.empty(0)
        threshold = 0.1
        # Keep values in b that are within 0.1 of any value in a
        mask = np.any(np.abs(timeline[:, None] - category_t) < threshold, axis=1)
        cat_metrics_t = np.concatenate((cat_metrics_t, timeline[mask]))
        for i in range(len(metrics_raw)):
            cat_metrics.append(metrics_raw[i][mask, 1])
            
        total_metrics.append(cat_metrics)
        total_metrics_t.append(cat_metrics_t)

    return total_metrics, total_metrics_t

def plot_violin(data, categories, title, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))  # bigger figure

    vp = ax.violinplot(data, showmeans=True, showextrema=False)

    for body in vp['bodies']:
        body.set_facecolor('#4C78A8')
        body.set_edgecolor('black')
        body.set_alpha(0.6)

    ax.set_xticks(np.arange(1, len(categories) + 1))
    ax.set_xticklabels(categories, rotation=30, ha='right')  # rotate + align

    ax.set_title(title)

    fig.tight_layout()

    plt.savefig(
        save_path / Path(f"{title}.png"),
        dpi=300,               # higher resolution
        bbox_inches='tight'    # prevents labels from being cut off
    )

    print(save_path / Path(f"{title}.png"))
    plt.close(fig)

def gen_violin_plot(df, save_path, metric):
    categories = df['Category'].unique().tolist()
    data = []
    for category in categories:
        category_data = df[df['Category'] == category]
        category_data = category_data[metric]
        data.append(category_data)
    
    plot_violin(
        data,
        categories,
        metric,
        save_path,
    )


def gen_dim_reduction_plot(df, save_path, title="pca_2d"):
    feature_cols = df.columns[:-1]
    category_col = df.columns[-1]

    points = df[feature_cols].to_numpy(dtype=float)
    categories = df[category_col].astype(str).to_numpy()
    valid_points = np.isfinite(points).all(axis=1)
    points = points[valid_points]
    categories = categories[valid_points]

    # points = points - points.mean(axis=0)
    points_tot = []
    _, _, vt = np.linalg.svd(points, full_matrices=False)
    vt_orth = null_space(vt[:2])
    points_tot.append(points @ vt[:2].T)
    points_tot.append(points @ vt_orth)
    points_tot.append(points @ np.asarray([vt_orth[:, 0], vt[:2].T[:, 1]]).T)
    points_tot.append(points @ np.asarray([vt[:2].T[:, 0], vt_orth[:, 1]]).T)

    for i, points_2d in enumerate(points_tot):
        unique_vals, idx = np.unique(categories, return_index=True)
        unique_categories = unique_vals[np.argsort(idx)]
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))

        fig, ax = plt.subplots()
        for category, color in zip(unique_categories, colors):
            category_mask = categories == category
            if category_mask.sum() > 800:
                true_indices = np.flatnonzero(category_mask)  # positions of True values
                remove_idx = np.random.choice(true_indices, size=800, replace=False)
                category_mask = np.full(category_mask.shape, False)
                category_mask[remove_idx] = True
            ax.scatter(
                points_2d[category_mask, 0],
                points_2d[category_mask, 1],
                s=SIZE,
                alpha=ALPHA_VALUE,
                edgecolors='none',
                linewidths=0,
                color=color,
                label=category,
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title)
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.xticks(rotation=30)
        plt.savefig(save_path / Path(f"{title}_{i}.png"), dpi=150)
        print(save_path / Path(f"{title}_{i}.png"))
        plt.close(fig)


def gen_dim_reduction_min_var_plot(df, save_path, title="pca_2d"):
    feature_cols = df.columns[:-1]
    category_col = df.columns[-1]

    points = df[feature_cols].to_numpy(dtype=float)
    categories = df[category_col].astype(str).to_numpy()
    valid_points = np.isfinite(points).all(axis=1)
    points = points[valid_points]
    categories = categories[valid_points]
    unique_vals, idx = np.unique(categories, return_index=True)
    unique_categories = unique_vals[np.argsort(idx)]

    category_perp = []
    for category in unique_categories:
        category_mask = categories == category
        category_points = points[category_mask]
        _, _, vt = np.linalg.svd(category_points, full_matrices=False)
        category_perp.append(vt[-2:])
        
    v1, v2 = np.zeros(vt[0].shape), np.zeros(vt[0].shape)
    for i in range(len(category_perp)):
        v1 = v1 + category_perp[i][0, :]
        v2 = v2 + category_perp[i][1, :]
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    points_tot = []
    _, _, vt = np.linalg.svd(points, full_matrices=False)
    vt_orth = null_space(vt[:2])
    points_tot.append(points @ np.asarray([v1, v2]).T)
    points_tot.append(points @ np.asarray([v1, vt[:2].T[:, 0]]).T)
    points_tot.append(points @ np.asarray([vt[:2].T[:, 0], v2]).T)

    for i, points_2d in enumerate(points_tot):
        unique_vals, idx = np.unique(categories, return_index=True)
        unique_categories = unique_vals[np.argsort(idx)]
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))

        fig, ax = plt.subplots()
        for category, color in zip(unique_categories, colors):
            category_mask = categories == category
            if category_mask.sum() > 800:
                true_indices = np.flatnonzero(category_mask)  # positions of True values
                remove_idx = np.random.choice(true_indices, size=800, replace=False)
                category_mask = np.full(category_mask.shape, False)
                category_mask[remove_idx] = True
                ax.scatter(
                    points_2d[category_mask, 0],
                    points_2d[category_mask, 1],
                    s=SIZE,
                    alpha=ALPHA_VALUE,
                    edgecolors='none',
                    linewidths=0,
                    color=color,
                    label=category,
                )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title)
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.xticks(rotation=30)
        plt.savefig(save_path / Path(f"{title}_perp_{i}.png"), dpi=150)
        print(save_path / Path(f"{title}_perp_{i}.png"))
        plt.close(fig)


def gen_dim_reduction_3d_plot(df, save_path, title="pca_3d"):
    feature_cols = df.columns[:-1]
    category_col = df.columns[-1]

    points = df[feature_cols].to_numpy(dtype=float)
    categories = df[category_col].astype(str).to_numpy()
    valid_points = np.isfinite(points).all(axis=1)
    points = points[valid_points]
    categories = categories[valid_points]

    points_tot = []
    # points = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(points, full_matrices=False)
    vt_orth = null_space(vt[:3])
    points_tot.append(points @ vt[:3].T)
    points_tot.append(points @ np.asarray([vt_orth[:, 0], vt[:3].T[:, 1], vt[:3].T[:, 2]]).T)
    points_tot.append(points @ np.asarray([vt[:3].T[:, 0], vt_orth[:, 0], vt[:3].T[:, 2]]).T)
    points_tot.append(points @ np.asarray([vt[:3].T[:, 0], vt[:3].T[:, 1], vt_orth[:, 0]]).T)

    unique_vals, idx = np.unique(categories, return_index=True)
    unique_categories = unique_vals[np.argsort(idx)]
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))

    for i, points_3d in enumerate(points_tot):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for category, color in zip(unique_categories, colors):
            category_mask = categories == category
            if category_mask.sum() > 800:
                true_indices = np.flatnonzero(category_mask)  # positions of True values
                remove_idx = np.random.choice(true_indices, size=800, replace=False)
                category_mask = np.full(category_mask.shape, False)
                category_mask[remove_idx] = True

            ax.scatter(
                points_3d[category_mask, 0],
                points_3d[category_mask, 1],
                points_3d[category_mask, 2],
                s=SIZE,
                alpha=ALPHA_VALUE,
                color=color,
                edgecolors='none',
                linewidths=0,
                label=category,
                depthshade=True
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(title)
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.xticks(rotation=30)
        plt.savefig(save_path / Path(f"{title}_{i}.png"), dpi=150)
        print(save_path / Path(f"{title}_{i}.png"))
        plt.close(fig)

def gen_dim_reduction_3d_plot_min_var(df, save_path, title="lda_3d"):
    feature_cols = df.columns[:-1]
    category_col = df.columns[-1]

    points = df[feature_cols].to_numpy(dtype=float)
    categories = df[category_col].astype(str).to_numpy()
    valid_points = np.isfinite(points).all(axis=1)
    points = points[valid_points]
    categories = categories[valid_points]

    unique_vals, idx = np.unique(categories, return_index=True)
    unique_categories = unique_vals[np.argsort(idx)]
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))

    point_mean = points.mean(axis=0)
    point_std = points.std(axis=0)
    point_std[point_std == 0] = 1
    #points = (points - point_mean) / point_std

    overall_mean = points.mean(axis=0)
    within_scatter = np.zeros((points.shape[1], points.shape[1]))
    between_scatter = np.zeros_like(within_scatter)
    for category in unique_categories:
        category_points = points[categories == category]
        category_mean = category_points.mean(axis=0)
        centered = category_points - category_mean
        within_scatter += centered.T @ centered
        mean_diff = category_mean - overall_mean
        between_scatter += len(category_points) * np.outer(mean_diff, mean_diff)

    reg = 1e-6 * np.trace(within_scatter) / max(1, within_scatter.shape[0])
    within_scatter = within_scatter + reg * np.eye(within_scatter.shape[0])
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(within_scatter) @ between_scatter)
    eigvals = eigvals.real
    eigvecs = eigvecs.real
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    n_lda_axes = min(3, points.shape[1], max(1, len(unique_categories) - 1))
    axes = eigvecs[:, :n_lda_axes].T

    if axes.shape[0] < 3:
        _, _, vt = np.linalg.svd(points, full_matrices=False)
        for pc_axis in vt:
            if axes.shape[0] == 3:
                break
            candidate = pc_axis.copy()
            for lda_axis in axes:
                candidate = candidate - np.dot(candidate, lda_axis) * lda_axis
            candidate_norm = np.linalg.norm(candidate)
            if candidate_norm > 1e-12:
                axes = np.vstack([axes, candidate / candidate_norm])

    points_3d = points @ axes[:3].T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for category, color in zip(unique_categories, colors):
        category_mask = categories == category
        if category_mask.sum() > 800:
            true_indices = np.flatnonzero(category_mask)  # positions of True values
            remove_idx = np.random.choice(true_indices, size=800, replace=False)
            category_mask = np.full(category_mask.shape, False)
            category_mask[remove_idx] = True

        ax.scatter(
            points_3d[category_mask, 0],
            points_3d[category_mask, 1],
            points_3d[category_mask, 2],
            s=SIZE,
            alpha=ALPHA_VALUE,
            color=color,
            edgecolors='none',
            linewidths=0,
            label=category,
            depthshade=True
        )

    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.set_zlabel("LD3")
    ax.set_title(title)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.xticks(rotation=30)
    plt.savefig(save_path / Path(f"{title}.png"), dpi=150)
    print(save_path / Path(f"{title}.png"))
    plt.show()
    plt.close(fig)


def gen_whole_session_plots(save_path, metrics_used, session_type, categories, timeline, categories_t, metrics_raw):
    
    for metric_name, metric in zip(metrics_used, metrics_raw):
        fig, ax = plt.subplots(figsize=(150, 6))  # bigger figure
        for i, category_t in enumerate(categories_t):
            color=f'C{i + 1}'
            for j, t in enumerate(category_t[::10]):
                if j == 0: plt.axvline(x=t, color=color, alpha=0.7, label=categories[i])
                else: plt.axvline(x=t, color=color, alpha=0.7)
        ax.plot(timeline, metric[:, 1], color='C0')
        ax.legend(loc='upper left')
        plt.savefig(os.path.join(save_path, 'whole_session_' + metric_name + '_' + session_type + '.png'))
        print(os.path.join(save_path, 'whole_session_' + metric_name + '_' + session_type + '.png'))
