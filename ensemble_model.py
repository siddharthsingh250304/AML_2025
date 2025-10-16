import lightgbm as lgb
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import faiss
import warnings
import gc
warnings.filterwarnings('ignore')

print(f"FAISS GPUs: {faiss.get_num_gpus()}")

# ============================================================================
# LOAD DATA
# ============================================================================
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
prices = train_df['price'].values

print(f"Train dataset: {len(train_df)} samples")
print(f"Test dataset:  {len(test_df)} samples")
print(f"Price stats: mean=${prices.mean():.2f}, median=${np.median(prices):.2f}")

# ============================================================================
# SPLIT
# ============================================================================
print(f"\n{'='*80}")
print("🔀 SPLIT: 90% Train | 10% Val")
print("="*80)

n_total = len(train_df)
n_train_final = int(0.90 * n_total)
n_val = n_total - n_train_final

train_final_idx = np.arange(0, n_train_final)
val_idx = np.arange(n_train_final, n_total)

print(f"Train: {len(train_final_idx)} samples")
print(f"Val:   {len(val_idx)} samples")

def smape_score(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0.01)
    return 100 * np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2))

# ============================================================================
# PRELOAD ALL EMBEDDINGS
# ============================================================================
print(f"\n{'='*80}")
print("📦 PRELOADING EMBEDDINGS")
print("="*80)

text_embedding_paths = {
    'bge-large-en': 'features/train/bge-large-en_features.npy',
    'distiluse-base-multilingual': 'features/train/distiluse-base-multilingual_features.npy',
    'all-mpnet-base-v2': 'features/train/all-mpnet-base-v2_features.npy',
    'nomic-embed-text-v1.5': 'features/train/nomic-embed-text-v1.5_features.npy',
    'mxbai-embed-large-v1': 'features/train/mxbai-embed-large-v1_features.npy',
    'snowflake-arctic-embed-l': 'features/train/snowflake-arctic-embed-l_features.npy',
    'paraphrase-MiniLM-L6-v2': 'features/train/paraphrase-MiniLM-L6-v2_features.npy',
    'all-MiniLM-L12-v2': 'features/train/all-MiniLM-L12-v2_features.npy',
    'clip_ft': 'features/train/ft_clip_70.npy',
    'distilbert_ft': 'features/train/distilbert_ft.npy',
    'distiluse_ft': 'features/train/distiluse_ft.npy',
    'distilbert_2field_ft': 'features/train/distilbert_2field_features.npy',
    'train_text_embeddings': 'features/train/train_text_embeddings.npy',
}

image_embedding_paths = {
    'clip-vit-large': 'features/train/clip-vit-large-patch14_features.npy',
    'siglip-base-patch16-224': 'features/train/siglip-base-patch16-224_features.npy',
    'nomic-embed-vision': 'features/train/nomic-embed-vision-v1.5_features.npy',
    'dinov2_ft': 'features/train/dinov2_ft_features.npy',
    'clip_ft_img': 'features/train/clip_vision_ft_train.npy',
}

text_embedding_paths_test = {k: v.replace('/train/', '/test/') for k, v in text_embedding_paths.items()}
text_embedding_paths_test['nomic-embed-vision'] = 'features/test/nomic-embed-vision-v1.5_features_new.npy'

image_embedding_paths_test = {k: v.replace('/train/', '/test/') for k, v in image_embedding_paths.items()}
image_embedding_paths_test['nomic-embed-vision'] = 'features/test/nomic-embed-vision-v1.5_features_new.npy'

# Load train embeddings
print("Loading TRAIN...")
train_embeddings_text = {name: np.load(path).astype('float32') for name, path in text_embedding_paths.items()}
train_embeddings_image = {name: np.load(path).astype('float32') for name, path in image_embedding_paths.items()}

# Load test embeddings
print("Loading TEST...")
test_embeddings_text = {name: np.load(path).astype('float32') for name, path in text_embedding_paths_test.items()}
test_embeddings_image = {name: np.load(path).astype('float32') for name, path in image_embedding_paths_test.items()}

print(f"✅ All embeddings loaded")

# ============================================================================
# BATCHED KNN FEATURE GENERATION
# ============================================================================

def generate_knn_features_batched(emb, prices, knn_pool_idx, query_idx, k, batch_size=25000):
    """Generate KNN features with BATCHED queries to save GPU memory"""
    
    knn_emb = emb[knn_pool_idx].copy()
    knn_prices = prices[knn_pool_idx]
    d = knn_emb.shape[1]
    
    # Normalize KNN pool once
    faiss.normalize_L2(knn_emb)
    
    # Build index once
    index = faiss.IndexFlatIP(d)
    
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        res.setTempMemory(1536 * 1024 * 1024)  # 1.5GB temp memory
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    index.add(knn_emb)
    
    # Process queries in BATCHES
    n_queries = len(query_idx)
    n_batches = int(np.ceil(n_queries / batch_size))
    
    all_features = []
    
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, n_queries)
        batch_query_idx = query_idx[start:end]
        
        query_emb_batch = emb[batch_query_idx].copy()
        faiss.normalize_L2(query_emb_batch)
        
        # Search
        dist, idx = index.search(query_emb_batch, k+1)
        
        # Leave-one-out handling
        if np.isin(batch_query_idx, knn_pool_idx).any():
            dist = dist[:, 1:k+1]
            idx = idx[:, 1:k+1]
        else:
            dist = dist[:, :k]
            idx = idx[:, :k]
        
        neighbor_prices = knn_prices[idx]
        weights = 1.0 / (dist + 1e-6)
        
        # Vectorized features
        price_mean = neighbor_prices.mean(axis=1)
        price_std = neighbor_prices.std(axis=1)
        price_min = neighbor_prices.min(axis=1)
        price_max = neighbor_prices.max(axis=1)
        price_median = np.median(neighbor_prices, axis=1)
        price_weighted_mean = np.average(neighbor_prices, axis=1, weights=weights)
        price_p25 = np.percentile(neighbor_prices, 25, axis=1)
        price_p75 = np.percentile(neighbor_prices, 75, axis=1)
        price_range = price_max - price_min
        price_iqr = price_p75 - price_p25
        price_cv = price_std / (price_mean + 1e-6)
        price_weighted_sum = (weights * neighbor_prices).sum(axis=1) / weights.sum(axis=1)
        
        k_actual = min(k, neighbor_prices.shape[1])
        price_top5_mean = neighbor_prices[:, :min(5, k_actual)].mean(axis=1)
        price_top10_mean = neighbor_prices[:, :min(10, k_actual)].mean(axis=1)
        
        price_ratios_mean = neighbor_prices / (price_mean[:, np.newaxis] + 1e-6)
        price_ratio_min = price_ratios_mean.min(axis=1)
        price_ratio_max = price_ratios_mean.max(axis=1)
        price_ratio_std = price_ratios_mean.std(axis=1)
        
        dist_mean = dist.mean(axis=1, keepdims=True)
        dist_max = dist.max(axis=1, keepdims=True)
        dist_normalized_by_mean = dist / (dist_mean + 1e-6)
        dist_normalized_by_max = dist / (dist_max + 1e-6)
        dist_norm_mean_stat = dist_normalized_by_mean.mean(axis=1)
        dist_norm_max_stat = dist_normalized_by_max.mean(axis=1)
        dist_relative_spread = dist.std(axis=1) / (dist_mean.flatten() + 1e-6)
        
        mid = k // 2
        price_top = neighbor_prices[:, :mid].mean(axis=1)
        price_bottom = neighbor_prices[:, mid:].mean(axis=1)
        price_momentum = price_top - price_bottom
        price_momentum_ratio = price_top / (price_bottom + 1e-6)
        
        positions = np.arange(k).reshape(1, -1)
        price_gradient = np.array([np.polyfit(positions.flatten(), neighbor_prices[i], 1)[0] 
                                   for i in range(len(neighbor_prices))])
        
        batch_features = np.column_stack([
            neighbor_prices, dist,
            price_mean, price_std, price_min, price_max, price_median,
            price_weighted_mean, price_p25, price_p75, price_range, price_iqr,
            price_cv, price_weighted_sum, price_top5_mean, price_top10_mean,
            price_ratio_min, price_ratio_max, price_ratio_std,
            dist_norm_mean_stat, dist_norm_max_stat, dist_relative_spread,
            price_momentum, price_momentum_ratio, price_gradient,
        ])
        
        all_features.append(batch_features)
        
        del query_emb_batch, dist, idx, neighbor_prices, batch_features
        gc.collect()
    
    # Cleanup
    del knn_emb, index
    if faiss.get_num_gpus() > 0:
        del res
    gc.collect()
    
    return np.vstack(all_features)

def add_consensus_features(features_per_model, k):
    price_means = np.column_stack([feat[:, 2*k] for feat in features_per_model])
    price_medians = np.column_stack([feat[:, 2*k+4] for feat in features_per_model])
    price_weighted_means = np.column_stack([feat[:, 2*k+5] for feat in features_per_model])
    
    consensus_mean = price_means.mean(axis=1)
    consensus_std = price_means.std(axis=1)
    consensus_median = price_medians.mean(axis=1)
    consensus_weighted_mean = price_weighted_means.mean(axis=1)
    mean_median_ratio = consensus_mean / (consensus_median + 1e-6)
    weighted_simple_ratio = consensus_weighted_mean / (consensus_mean + 1e-6)
    price_agreement = 1.0 - (price_means.std(axis=1) / (price_means.mean(axis=1) + 1e-6))
    
    return np.column_stack([
        consensus_mean, consensus_std, consensus_median, consensus_weighted_mean,
        mean_median_ratio, weighted_simple_ratio, price_agreement,
    ])

def add_modality_consensus_features(text_features, image_features, k):
    text_means = np.column_stack([feat[:, 2*k] for feat in text_features])
    text_consensus_mean = text_means.mean(axis=1)
    text_consensus_std = text_means.std(axis=1)
    
    image_means = np.column_stack([feat[:, 2*k] for feat in image_features])
    image_consensus_mean = image_means.mean(axis=1)
    image_consensus_std = image_means.std(axis=1)
    
    text_image_ratio = text_consensus_mean / (image_consensus_mean + 1e-6)
    text_image_diff = text_consensus_mean - image_consensus_mean
    modality_agreement = 1.0 - np.abs(text_image_diff) / (text_consensus_mean + image_consensus_mean + 1e-6)
    
    return np.column_stack([
        text_consensus_mean, text_consensus_std,
        image_consensus_mean, image_consensus_std,
        text_image_ratio, text_image_diff, modality_agreement,
    ])

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

lgb_params = {
    'objective': 'regression',
    'metric': 'mse',
    'learning_rate': 0.02,
    'max_depth': 9,
    'num_leaves': 120,
    'min_child_samples': 50,
    'subsample': 0.75,
    'colsample_bytree': 0.65,
    'reg_alpha': 1.2,
    'reg_lambda': 2.5,
    'n_estimators': 2000,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1,
}

cat_params = {
    'iterations': 2000,
    'learning_rate': 0.02,
    'depth': 9,
    'l2_leaf_reg': 5,
    'loss_function': 'Quantile:alpha=0.5',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': False,
    'early_stopping_rounds': 200,
    'thread_count': -1,
}

# ============================================================================
# TRAINING LOOP
# ============================================================================

print(f"\n{'='*80}")
print("🔄 TRAINING (BATCHED KNN)")
print("="*80)

BATCH_SIZE = 25000
k_values = [5, 10]
knn_pool_idx = np.arange(len(train_df))

y_train = np.log1p(prices[train_final_idx])
y_val = np.log1p(prices[val_idx])
y_val_original = prices[val_idx]

final_models = {}

for k in k_values:
    print(f"\n{'─'*80}")
    print(f"K = {k}")
    print(f"{'─'*80}")
    
    print("Generating features (batched KNN)...")
    
    text_feats_train = []
    text_feats_val = []
    
    for name, emb in train_embeddings_text.items():
        print(f"  Text: {name}...", end=' ', flush=True)
        train_feat = generate_knn_features_batched(emb, prices, knn_pool_idx, train_final_idx, k, BATCH_SIZE)
        val_feat = generate_knn_features_batched(emb, prices, knn_pool_idx, val_idx, k, BATCH_SIZE)
        text_feats_train.append(train_feat)
        text_feats_val.append(val_feat)
        print("✓")
    
    image_feats_train = []
    image_feats_val = []
    
    for name, emb in train_embeddings_image.items():
        print(f"  Image: {name}...", end=' ', flush=True)
        train_feat = generate_knn_features_batched(emb, prices, knn_pool_idx, train_final_idx, k, BATCH_SIZE)
        val_feat = generate_knn_features_batched(emb, prices, knn_pool_idx, val_idx, k, BATCH_SIZE)
        image_feats_train.append(train_feat)
        image_feats_val.append(val_feat)
        print("✓")
    
    all_feats_train = text_feats_train + image_feats_train
    all_feats_val = text_feats_val + image_feats_val
    
    train_consensus = add_consensus_features(all_feats_train, k)
    val_consensus = add_consensus_features(all_feats_val, k)
    
    train_modality = add_modality_consensus_features(text_feats_train, image_feats_train, k)
    val_modality = add_modality_consensus_features(text_feats_val, image_feats_val, k)
    
    X_train = np.concatenate(all_feats_train + [train_consensus, train_modality], axis=1)
    X_val = np.concatenate(all_feats_val + [val_consensus, val_modality], axis=1)
    
    print(f"Features: {X_train.shape[1]}")
    
    # Train LightGBM
    print("Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    
    importance = lgb_model.feature_importances_
    importance_threshold = np.percentile(importance, 40)
    important_features = importance > importance_threshold
    
    print(f"Feature selection: {important_features.sum()}/{len(important_features)}")
    
    X_train_sel = X_train[:, important_features]
    X_val_sel = X_val[:, important_features]
    
    lgb_final = lgb.LGBMRegressor(**lgb_params)
    lgb_final.fit(X_train_sel, y_train, eval_set=[(X_val_sel, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    
    lgb_pred = np.expm1(lgb_final.predict(X_val_sel))
    
    # CatBoost
    print("Training CatBoost...")
    cat_final = CatBoostRegressor(**cat_params)
    cat_final.fit(X_train_sel, y_train, eval_set=(X_val_sel, y_val))
    
    cat_pred = np.expm1(cat_final.predict(X_val_sel))
    
    blended = 0.2 * lgb_pred + 0.8 * cat_pred
    val_smape = smape_score(y_val_original, blended)
    
    print(f"Val SMAPE: {val_smape:.2f}%")
    
    final_models[k] = {
        'lgb': lgb_final,
        'cat': cat_final,
        'features': important_features,
    }

# ============================================================================
# TEST PREDICTIONS (BATCHED KNN)
# ============================================================================

print(f"\n{'='*80}")
print("🎯 TEST PREDICTIONS (BATCHED KNN)")
print("="*80)

test_predictions = {}

for k in k_values:
    print(f"\nK={k}")
    
    text_feats_test = []
    for name, emb_train in train_embeddings_text.items():
        print(f"  Text: {name}...", end=' ', flush=True)
        emb_test = test_embeddings_text[name]
        combined = np.vstack([emb_train, emb_test])
        combined_prices = np.concatenate([prices, np.zeros(len(emb_test))])
        test_query_idx = np.arange(len(emb_train), len(emb_train) + len(emb_test))
        knn_pool = np.arange(len(emb_train))
        feats = generate_knn_features_batched(combined, combined_prices, knn_pool, test_query_idx, k, BATCH_SIZE)
        text_feats_test.append(feats)
        print("✓")
    
    image_feats_test = []
    for name, emb_train in train_embeddings_image.items():
        print(f"  Image: {name}...", end=' ', flush=True)
        emb_test = test_embeddings_image[name]
        combined = np.vstack([emb_train, emb_test])
        combined_prices = np.concatenate([prices, np.zeros(len(emb_test))])
        test_query_idx = np.arange(len(emb_train), len(emb_train) + len(emb_test))
        knn_pool = np.arange(len(emb_train))
        feats = generate_knn_features_batched(combined, combined_prices, knn_pool, test_query_idx, k, BATCH_SIZE)
        image_feats_test.append(feats)
        print("✓")
    
    all_feats_test = text_feats_test + image_feats_test
    test_consensus = add_consensus_features(all_feats_test, k)
    test_modality = add_modality_consensus_features(text_feats_test, image_feats_test, k)
    
    X_test = np.concatenate(all_feats_test + [test_consensus, test_modality], axis=1)
    X_test_sel = X_test[:, final_models[k]['features']]
    
    lgb_pred = np.expm1(final_models[k]['lgb'].predict(X_test_sel))
    cat_pred = np.expm1(final_models[k]['cat'].predict(X_test_sel))
    blended = 0.2 * lgb_pred + 0.8 * cat_pred
    
    test_predictions[k] = blended

# ============================================================================
# ENSEMBLE & SAVE
# ============================================================================

print(f"\n{'='*80}")
print("🏆 FINAL ENSEMBLE")
print("="*80)

best_weights = [0.40, 0.60]
preds_list = [test_predictions[k] for k in k_values]
final_test_preds = np.average(preds_list, axis=0, weights=best_weights)

submission = pd.DataFrame({
    'sample_id': test_df['sample_id'].values,
    'pred': final_test_preds
})

submission.to_csv('submission.csv', index=False)
print(f"✅ Submission saved")
print(f"Range: ${final_test_preds.min():.2f} - ${final_test_preds.max():.2f}")
print(f"Mean: ${final_test_preds.mean():.2f}")
print(submission.head())
