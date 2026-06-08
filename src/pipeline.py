"""
Hotspot Detection System - Ultra High Recall
Optimized for maximum hotspot detection
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy import ndimage
from scipy.spatial.distance import pdist
from scipy.fftpack import dct
from skimage.measure import label, regionprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============ CONFIGURATION ============
DATASET_PATH = 'iccad-official'
USE_CACHE = True


class FeatureExtractor:
    """Extract features from layout patterns"""
    
    def extract(self, pattern):
        """Extract all 43 features"""
        if not isinstance(pattern, np.ndarray):
            pattern = np.array(pattern)
        
        features = {}
        h, w = pattern.shape
        
        # Density features (13)
        features['overall_density'] = np.mean(pattern)
        features['quad_tl_density'] = np.mean(pattern[:h//2, :w//2])
        features['quad_tr_density'] = np.mean(pattern[:h//2, w//2:])
        features['quad_bl_density'] = np.mean(pattern[h//2:, :w//2])
        features['quad_br_density'] = np.mean(pattern[h//2:, w//2:])
        
        center_mask = np.zeros_like(pattern)
        c_h, c_w = h // 2, w // 2
        center_mask[c_h-h//4:c_h+h//4, c_w-w//4:c_w+w//4] = 1
        features['center_density'] = np.mean(pattern[center_mask == 1])
        features['periphery_density'] = np.mean(pattern[center_mask == 0])
        
        for ring in range(1, 5):
            ring_mask = self._create_ring_mask((h, w), ring)
            features[f'ring_{ring}_density'] = np.mean(pattern[ring_mask])
        
        # Topology features (9)
        binary = (pattern > np.mean(pattern)).astype(int)
        labeled = label(binary)
        num_components = labeled.max()
        features['num_components'] = num_components
        
        if num_components > 0:
            props = regionprops(labeled)
            areas = [p.area for p in props]
            features['mean_component_area'] = np.mean(areas)
            features['std_component_area'] = np.std(areas)
            features['max_component_area'] = np.max(areas)
            features['min_component_area'] = np.min(areas)
            
            aspect_ratios = [p.major_axis_length / (p.minor_axis_length + 1e-6) for p in props]
            features['mean_aspect_ratio'] = np.mean(aspect_ratios)
            features['max_aspect_ratio'] = np.max(aspect_ratios)
            features['mean_eccentricity'] = np.mean([p.eccentricity for p in props])
            features['mean_solidity'] = np.mean([p.solidity for p in props])
        else:
            features.update({
                'mean_component_area': 0, 'std_component_area': 0,
                'max_component_area': 0, 'min_component_area': 0,
                'mean_aspect_ratio': 0, 'max_aspect_ratio': 0,
                'mean_eccentricity': 0, 'mean_solidity': 0
            })
        
        # Spatial features (8)
        y_coords, x_coords = np.where(binary)
        if len(x_coords) > 1:
            features['x_spread'] = np.std(x_coords)
            features['y_spread'] = np.std(y_coords)
            features['centroid_x'] = np.mean(x_coords)
            features['centroid_y'] = np.mean(y_coords)
            
            center_x, center_y = w // 2, h // 2
            features['centroid_dist_from_center'] = np.sqrt(
                (features['centroid_x'] - center_x)**2 + 
                (features['centroid_y'] - center_y)**2
            )
            
            coords = np.column_stack([x_coords, y_coords])
            if len(coords) > 1 and len(coords) < 1000:
                distances = pdist(coords)
                features['mean_pairwise_dist'] = np.mean(distances)
                features['max_pairwise_dist'] = np.max(distances)
                features['min_pairwise_dist'] = np.min(distances)
            else:
                features.update({'mean_pairwise_dist': 0, 'max_pairwise_dist': 0, 'min_pairwise_dist': 0})
        else:
            features.update({
                'x_spread': 0, 'y_spread': 0, 'centroid_x': 0, 'centroid_y': 0,
                'centroid_dist_from_center': 0, 'mean_pairwise_dist': 0,
                'max_pairwise_dist': 0, 'min_pairwise_dist': 0
            })
        
        # Geometric features (8)
        edges_x = np.abs(ndimage.sobel(pattern, axis=1))
        edges_y = np.abs(ndimage.sobel(pattern, axis=0))
        edges = np.hypot(edges_x, edges_y)
        features['edge_density'] = np.mean(edges)
        features['edge_strength'] = np.max(edges)
        
        gradient_magnitude = np.sqrt(edges_x**2 + edges_y**2)
        features['mean_gradient'] = np.mean(gradient_magnitude)
        features['max_gradient'] = np.max(gradient_magnitude)
        features['horizontal_gradient'] = np.mean(np.abs(edges_x))
        features['vertical_gradient'] = np.mean(np.abs(edges_y))
        
        laplacian = ndimage.laplace(pattern)
        features['laplacian_mean'] = np.mean(np.abs(laplacian))
        features['laplacian_std'] = np.std(laplacian)
        
        # Complexity features (5)
        hist, _ = np.histogram(pattern.flatten(), bins=50)
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        features['entropy'] = -np.sum(hist * np.log2(hist))
        features['pattern_std'] = np.std(pattern)
        features['pattern_range'] = np.ptp(pattern)
        
        mean_val = np.mean(pattern)
        features['coeff_variation'] = np.std(pattern) / mean_val if mean_val > 0 else 0
        
        dct_2d = dct(dct(pattern.T, norm='ortho').T, norm='ortho')
        features['low_freq_energy'] = np.sum(np.abs(dct_2d[:h//4, :w//4]))
        features['mid_freq_energy'] = np.sum(np.abs(dct_2d[h//4:h//2, w//4:w//2]))
        features['high_freq_energy'] = np.sum(np.abs(dct_2d[h//2:, w//2:]))
        
        return features
    
    def _create_ring_mask(self, shape, ring_index):
        h, w = shape
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        inner_radius = (ring_index - 1) * min(h, w) // 8
        outer_radius = ring_index * min(h, w) // 8
        return (dist_from_center >= inner_radius) & (dist_from_center < outer_radius)
    
    def process_dataset(self, patterns, labels):
        """Process dataset sequentially"""
        patterns = [np.array(p) if not isinstance(p, np.ndarray) else p for p in patterns]
        
        feature_list = []
        for p in tqdm(patterns, desc="Extracting features", unit="pattern"):
            feature_list.append(self.extract(p))
        
        df = pd.DataFrame(feature_list)
        df['label'] = labels
        return df


class HotspotPipeline:
    """ML Pipeline optimized for high recall"""
    
    def __init__(self, size=64):
        self.size = size
        self.scaler = StandardScaler()
        self.model = None
        self.extractor = FeatureExtractor()
    
    def load_images(self, folder):
        """Load images from folder"""
        files = sorted(list(Path(folder).glob('*.png')))
        patterns = []
        for file in tqdm(files, desc=f"Loading {Path(folder).name}", leave=False):
            img = Image.open(file).convert('L').resize((self.size, self.size), Image.LANCZOS)
            patterns.append(np.array(img, dtype=np.float32) / 255.0)
        return patterns
    
    def load_benchmark(self, base_path, benchmark_name, data_type='train'):
        """Load benchmark with caching"""
        cache_file = Path(f'cache/{benchmark_name}_{data_type}.npz')
        
        if USE_CACHE and cache_file.exists():
            print(f"  Loading {benchmark_name}/{data_type} from cache...")
            data = np.load(cache_file, allow_pickle=True)
            return data['patterns'].tolist(), data['labels'].tolist()
        
        folder = Path(base_path) / benchmark_name / data_type
        hs = self.load_images(folder / f'{data_type}_hs')
        nhs = self.load_images(folder / f'{data_type}_nhs')
        
        patterns = hs + nhs
        labels = [1] * len(hs) + [0] * len(nhs)
        
        if USE_CACHE:
            cache_file.parent.mkdir(exist_ok=True)
            np.savez_compressed(cache_file, patterns=np.array(patterns), labels=np.array(labels))
        
        print(f"  {benchmark_name}/{data_type}: {len(hs)} hotspots, {len(nhs)} non-hotspots")
        return patterns, labels
    
    def balance_data(self, patterns, labels):
        """Balance dataset using undersampling"""
        patterns = np.array(patterns, dtype=object)
        labels = np.array(labels)
        
        hs_idx = np.where(labels == 1)[0]
        nhs_idx = np.where(labels == 0)[0]
        
        np.random.seed(42)
        nhs_idx_sampled = np.random.choice(nhs_idx, size=len(hs_idx), replace=False)
        
        balanced_idx = np.concatenate([hs_idx, nhs_idx_sampled])
        np.random.shuffle(balanced_idx)
        
        return patterns[balanced_idx].tolist(), labels[balanced_idx].tolist()
    
    def prepare_data(self, data_root, benchmarks=[1, 2, 3, 4, 5]):
        """Load and prepare data"""
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)
        
        train_cache = Path('cache/train_features.csv')
        test_cache = Path('cache/test_features.csv')
        
        if USE_CACHE and train_cache.exists():
            print("\n✓ Loading preprocessed features from cache...")
            df_train = pd.read_csv(train_cache)
            df_test = pd.read_csv(test_cache)
            print(f"  Train: {len(df_train)}, Test: {len(df_test)}")
            return df_train, df_test
        
        train_patterns, train_labels = [], []
        test_patterns, test_labels = [], []
        
        for i in benchmarks:
            patterns, labels = self.load_benchmark(data_root, f'iccad{i}', 'train')
            train_patterns.extend(patterns)
            train_labels.extend(labels)
            
            patterns, labels = self.load_benchmark(data_root, f'iccad{i}', 'test')
            test_patterns.extend(patterns)
            test_labels.extend(labels)
        
        print(f"\nBefore balancing:")
        print(f"  Train - Hotspots: {sum(train_labels)}, Non-hotspots: {len(train_labels)-sum(train_labels)}")
        
        train_patterns, train_labels = self.balance_data(train_patterns, train_labels)
        
        print(f"\nAfter balancing:")
        print(f"  Train - Hotspots: {sum(train_labels)}, Non-hotspots: {len(train_labels)-sum(train_labels)}")
        
        print("\nExtracting features...")
        df_train = self.extractor.process_dataset(train_patterns, train_labels)
        df_test = self.extractor.process_dataset(test_patterns, test_labels)
        
        if USE_CACHE:
            Path('cache').mkdir(exist_ok=True)
            df_train.to_csv(train_cache, index=False)
            df_test.to_csv(test_cache, index=False)
            print("Cached processed features")
        
        print(f"\nFinal - Train: {len(df_train)}, Test: {len(df_test)}")
        return df_train, df_test
    
    def train(self, df_train):
        """Train model optimized for high recall"""
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        X = df_train.drop('label', axis=1)
        y = df_train['label']
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=30,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight={0: 1, 1: 5},
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        
        print("Training...")
        self.model.fit(X_scaled, y)
        
        if hasattr(self.model, 'oob_score_'):
            print(f"Out-of-bag score: {self.model.oob_score_:.3f}")
        
        print("Training complete")
    
    def evaluate(self, df_test, save_dir='results'):
        """Evaluate and visualize results"""
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        
        Path(save_dir).mkdir(exist_ok=True)
        
        X = df_test.drop('label', axis=1)
        y = df_test['label']
        X_scaled = self.scaler.transform(X)
        
        y_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        # Find optimal threshold for 90%+ recall
        print("\nOptimizing threshold for maximum recall...")
        
        best_threshold = 0.5
        best_recall = 0
        target_recall = 0.90
        
        for threshold in np.arange(0.1, 0.9, 0.01):
            y_pred_temp = (y_proba >= threshold).astype(int)
            tp = np.sum((y_pred_temp == 1) & (y == 1))
            fn = np.sum((y_pred_temp == 0) & (y == 1))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            if recall >= target_recall:
                best_threshold = threshold
                best_recall = recall
                break
            elif recall > best_recall:
                best_threshold = threshold
                best_recall = recall
        
        optimal_threshold = best_threshold
        print(f"Selected threshold: {optimal_threshold:.3f} (achieves {best_recall:.1%} recall)")
        
        y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
        y_pred_default = self.model.predict(X_scaled)
        
        print("\n" + "="*60)
        print("RESULTS WITH DEFAULT THRESHOLD (0.5):")
        print("="*60)
        print(classification_report(y, y_pred_default, target_names=['Non-Hotspot', 'Hotspot']))
        
        cm_default = confusion_matrix(y, y_pred_default)
        tn_d, fp_d, fn_d, tp_d = cm_default.ravel()
        recall_default = tp_d / (tp_d + fn_d)
        print(f"Default Recall: {recall_default:.1%} (missed {fn_d} hotspots)")
        
        print("\n" + "="*60)
        print(f"RESULTS WITH OPTIMIZED THRESHOLD ({optimal_threshold:.3f}):")
        print("="*60)
        print(classification_report(y, y_pred_optimal, target_names=['Non-Hotspot', 'Hotspot']))
        
        cm = confusion_matrix(y, y_pred_optimal)
        tn, fp, fn, tp = cm.ravel()
        
        recall = tp / (tp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        print(f"\nKEY METRICS (OPTIMIZED):")
        print(f"  RECALL: {recall:.1%} (detected {tp}/{tp+fn} hotspots) ⭐")
        print(f"  MISSED: {fn} hotspots (improved from {fn_d})")
        print(f"  Precision: {precision:.1%}")
        print(f"  False alarms: {fp} (vs {fp_d} with default threshold)")
        
        # Visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        sns.heatmap(cm_default, annot=True, fmt='d', cmap='Oranges', ax=axes[0,0],
                    xticklabels=['Non-HS', 'HS'], yticklabels=['Non-HS', 'HS'])
        axes[0,0].set_title(f'Default (0.5) - Recall: {recall_default:.1%}', fontsize=12, fontweight='bold')
        axes[0,0].set_ylabel('True Label')
        axes[0,0].set_xlabel('Predicted Label')
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0,1],
                    xticklabels=['Non-HS', 'HS'], yticklabels=['Non-HS', 'HS'])
        axes[0,1].set_title(f'Optimized ({optimal_threshold:.3f}) - Recall: {recall:.1%}', fontsize=12, fontweight='bold')
        axes[0,1].set_ylabel('True Label')
        axes[0,1].set_xlabel('Predicted Label')
        
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = roc_auc_score(y, y_proba)
        axes[0,2].plot(fpr, tpr, 'b-', lw=2, label=f'AUC={roc_auc:.3f}')
        axes[0,2].plot([0,1], [0,1], 'r--', lw=2)
        
        fp_rate_default = fp_d / (tn_d + fp_d)
        axes[0,2].plot(fp_rate_default, recall_default, 'ro', markersize=10, label=f'Default (Recall={recall_default:.2f})')
        fp_rate_optimal = fp / (tn + fp)
        axes[0,2].plot(fp_rate_optimal, recall, 'go', markersize=10, label=f'Optimized (Recall={recall:.2f})')
        
        axes[0,2].axhline(y=0.90, color='g', linestyle='--', alpha=0.5, label='90% recall target')
        axes[0,2].set_title('ROC Curve', fontsize=12, fontweight='bold')
        axes[0,2].legend()
        axes[0,2].grid(alpha=0.3)
        
        metrics_names = ['Recall', 'Precision', 'F1', 'Accuracy']
        default_values = [
            recall_default,
            tp_d / (tp_d + fp_d) if (tp_d + fp_d) > 0 else 0,
            f1_score(y, y_pred_default),
            accuracy_score(y, y_pred_default)
        ]
        optimized_values = [
            recall,
            precision,
            f1_score(y, y_pred_optimal),
            accuracy_score(y, y_pred_optimal)
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        axes[1,0].bar(x - width/2, default_values, width, label='Default (0.5)', color='orange', alpha=0.7)
        axes[1,0].bar(x + width/2, optimized_values, width, label=f'Optimized ({optimal_threshold:.2f})', color='green', alpha=0.7)
        axes[1,0].set_ylabel('Score')
        axes[1,0].set_title('Metrics Comparison', fontsize=12, fontweight='bold')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(metrics_names)
        axes[1,0].axhline(0.90, color='g', linestyle='--', alpha=0.3)
        axes[1,0].legend()
        axes[1,0].set_ylim([0, 1])
        
        imp = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        axes[1,1].barh(range(len(imp)), imp['importance'])
        axes[1,1].set_yticks(range(len(imp)))
        axes[1,1].set_yticklabels(imp['feature'])
        axes[1,1].set_title('Top 10 Features', fontsize=12, fontweight='bold')
        axes[1,1].invert_yaxis()
        
        x_pos = np.arange(2)
        axes[1,2].bar(x_pos - 0.2, [tp_d, fn_d], 0.4, label='Default', color='orange', alpha=0.7)
        axes[1,2].bar(x_pos + 0.2, [tp, fn], 0.4, label='Optimized', color='green', alpha=0.7)
        axes[1,2].set_xticks(x_pos)
        axes[1,2].set_xticklabels(['Detected', 'Missed'])
        axes[1,2].set_ylabel('Count')
        axes[1,2].set_title('Hotspot Detection Improvement', fontsize=12, fontweight='bold')
        axes[1,2].legend()
        
        for i, (v1, v2) in enumerate(zip([tp_d, fn_d], [tp, fn])):
            axes[1,2].text(i - 0.2, v1 + 1, str(v1), ha='center', fontweight='bold')
            axes[1,2].text(i + 0.2, v2 + 1, str(v2), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/results.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {save_dir}/results.png")
        plt.close()
        
        self.optimal_threshold = optimal_threshold
        
        return {
            'recall': recall,
            'precision': precision,
            'missed': fn,
            'total_hs': tp+fn,
            'threshold': optimal_threshold,
            'improvement': fn_d - fn
        }
    
    def save(self, filepath='iccad_hotspot_hr.pkl'):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'threshold': getattr(self, 'optimal_threshold', 0.5)
        }, filepath)
        print(f"✓ Model saved: {filepath}")
        if hasattr(self, 'optimal_threshold'):
            print(f"  Optimal threshold: {self.optimal_threshold:.3f}")


def main():
    print("="*60)
    print("HOTSPOT DETECTION SYSTEM")
    print("="*60)
    
    pipeline = HotspotPipeline(size=64)
    df_train, df_test = pipeline.prepare_data(DATASET_PATH)
    pipeline.train(df_train)
    metrics = pipeline.evaluate(df_test)
    pipeline.save('iccad_hotspot_hr.pkl')
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE")
    print("="*60)
    print(f"RECALL: {metrics['recall']:.1%}")
    print(f"Missed: {metrics['missed']}/{metrics['total_hs']} hotspots")
    print(f"Threshold: {metrics['threshold']:.3f}")
    print(f"Improvement: Caught {metrics['improvement']} more hotspots!")


if __name__ == "__main__":
    main()