import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy import ndimage
from scipy.spatial.distance import pdist
from scipy.fftpack import dct
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')


def extract_features(pattern):
    """Extract 43 features from pattern - EXACT names from training"""
    h, w = pattern.shape
    features = {}
    
    # Density (13)
    features['overall_density'] = np.mean(pattern)
    features['quad_tl_density'] = np.mean(pattern[:h//2, :w//2])
    features['quad_tr_density'] = np.mean(pattern[:h//2, w//2:])
    features['quad_bl_density'] = np.mean(pattern[h//2:, :w//2])
    features['quad_br_density'] = np.mean(pattern[h//2:, w//2:])
    
    center_mask = np.zeros_like(pattern)
    c_h, c_w = h//2, w//2
    center_mask[c_h-h//4:c_h+h//4, c_w-w//4:c_w+w//4] = 1
    features['center_density'] = np.mean(pattern[center_mask == 1])
    features['periphery_density'] = np.mean(pattern[center_mask == 0])
    
    for ring in range(1, 5):
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - w//2)**2 + (y - h//2)**2)
        mask = (dist >= (ring-1)*min(h,w)//8) & (dist < ring*min(h,w)//8)
        features[f'ring_{ring}_density'] = np.mean(pattern[mask])
    
    # Topology (9)
    binary = (pattern > np.mean(pattern)).astype(int)
    labeled = label(binary)
    features['num_components'] = labeled.max()
    
    if labeled.max() > 0:
        props = regionprops(labeled)
        areas = [p.area for p in props]
        features['mean_component_area'] = np.mean(areas)
        features['std_component_area'] = np.std(areas)
        features['max_component_area'] = np.max(areas)
        features['min_component_area'] = np.min(areas)
        features['mean_aspect_ratio'] = np.mean([p.major_axis_length/(p.minor_axis_length+1e-6) for p in props])
        features['max_aspect_ratio'] = np.max([p.major_axis_length/(p.minor_axis_length+1e-6) for p in props])
        features['mean_eccentricity'] = np.mean([p.eccentricity for p in props])
        features['mean_solidity'] = np.mean([p.solidity for p in props])
    else:
        features['mean_component_area'] = 0
        features['std_component_area'] = 0
        features['max_component_area'] = 0
        features['min_component_area'] = 0
        features['mean_aspect_ratio'] = 0
        features['max_aspect_ratio'] = 0
        features['mean_eccentricity'] = 0
        features['mean_solidity'] = 0
    
    # Spatial (8)
    y_coords, x_coords = np.where(binary)
    if len(x_coords) > 1:
        features['x_spread'] = np.std(x_coords)
        features['y_spread'] = np.std(y_coords)
        features['centroid_x'] = np.mean(x_coords)
        features['centroid_y'] = np.mean(y_coords)
        features['centroid_dist_from_center'] = np.sqrt((np.mean(x_coords)-w//2)**2 + (np.mean(y_coords)-h//2)**2)
        if len(x_coords) < 1000:
            dists = pdist(np.column_stack([x_coords, y_coords]))
            features['mean_pairwise_dist'] = np.mean(dists)
            features['max_pairwise_dist'] = np.max(dists)
            features['min_pairwise_dist'] = np.min(dists)
        else:
            features['mean_pairwise_dist'] = 0
            features['max_pairwise_dist'] = 0
            features['min_pairwise_dist'] = 0
    else:
        features['x_spread'] = 0
        features['y_spread'] = 0
        features['centroid_x'] = 0
        features['centroid_y'] = 0
        features['centroid_dist_from_center'] = 0
        features['mean_pairwise_dist'] = 0
        features['max_pairwise_dist'] = 0
        features['min_pairwise_dist'] = 0
    
    # Geometric (8)
    edges_x = np.abs(ndimage.sobel(pattern, axis=1))
    edges_y = np.abs(ndimage.sobel(pattern, axis=0))
    edges = np.hypot(edges_x, edges_y)
    features['edge_density'] = np.mean(edges)
    features['edge_strength'] = np.max(edges)
    features['mean_gradient'] = np.mean(np.sqrt(edges_x**2 + edges_y**2))
    features['max_gradient'] = np.max(np.sqrt(edges_x**2 + edges_y**2))
    features['horizontal_gradient'] = np.mean(np.abs(edges_x))
    features['vertical_gradient'] = np.mean(np.abs(edges_y))
    laplacian = ndimage.laplace(pattern)
    features['laplacian_mean'] = np.mean(np.abs(laplacian))
    features['laplacian_std'] = np.std(laplacian)
    
    # Complexity (12)
    hist, _ = np.histogram(pattern.flatten(), bins=50)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    features['entropy'] = -np.sum(hist * np.log2(hist))
    features['pattern_std'] = np.std(pattern)
    features['pattern_range'] = np.ptp(pattern)
    features['coeff_variation'] = np.std(pattern) / np.mean(pattern) if np.mean(pattern) > 0 else 0
    dct_2d = dct(dct(pattern.T, norm='ortho').T, norm='ortho')
    features['low_freq_energy'] = np.sum(np.abs(dct_2d[:h//4, :w//4]))
    features['mid_freq_energy'] = np.sum(np.abs(dct_2d[h//4:h//2, w//4:w//2]))
    features['high_freq_energy'] = np.sum(np.abs(dct_2d[h//2:, w//2:]))
    
    return features


def load_test_data(base_path='iccad-official', n_samples=20):
    """Load small test subset"""
    hs_path = Path(base_path) / 'iccad1/test/test_hs'
    nhs_path = Path(base_path) / 'iccad1/test/test_nhs'
    
    patterns, labels, names = [], [], []
    
    for file in sorted(hs_path.glob('*.png'))[:n_samples//2]:
        img = Image.open(file).convert('L').resize((64, 64), Image.LANCZOS)
        patterns.append(np.array(img, dtype=np.float32) / 255.0)
        labels.append(1)
        names.append(file.name)
    
    for file in sorted(nhs_path.glob('*.png'))[:n_samples//2]:
        img = Image.open(file).convert('L').resize((64, 64), Image.LANCZOS)
        patterns.append(np.array(img, dtype=np.float32) / 255.0)
        labels.append(0)
        names.append(file.name)
    
    return patterns, labels, names


def test_model(model_path='iccad_hotspot_hr.pkl', n_samples=20):
    """Test model and show results"""
    print("="*60)
    print("HOTSPOT DETECTION TEST")
    print("="*60)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    data = joblib.load(model_path)
    model, scaler, threshold = data['model'], data['scaler'], data.get('threshold', 0.5)
    print(f"✓ Loaded (threshold: {threshold:.3f})")
    
    # Load test data
    print(f"\nLoading {n_samples} test samples...")
    patterns, labels, names = load_test_data(n_samples=n_samples)
    print(f"✓ Loaded {sum(labels)} hotspots, {len(labels)-sum(labels)} non-hotspots")
    
    # Extract features and predict
    print("\nExtracting features and predicting...")
    features = [extract_features(p) for p in patterns]
    X = scaler.transform(pd.DataFrame(features))
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    # Calculate metrics
    labels = np.array(labels)
    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    
    acc = (tp + tn) / len(labels)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    # Display results
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    print(f"\n              Predicted")
    print(f"            NHS    HS")
    print(f"     NHS  [{tn:4d}] [{fp:4d}]")
    print(f"True")
    print(f"      HS  [{fn:4d}] [{tp:4d}]")
    
    print("\n" + "="*60)
    print("METRICS")
    print("="*60)
    print(f"\n  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Show predictions
    results = pd.DataFrame({'name': names, 'true': labels, 'prob': probs, 'pred': preds})
    results = results.sort_values('prob', ascending=False)
    
    print("\n" + "="*60)
    print("TOP 5 HIGHEST RISK")
    print("="*60)
    for _, row in results.head(5).iterrows():
        check = "✓" if row['pred'] == row['true'] else "✗"
        print(f"{check} {row['name']}: P={row['prob']:.3f} | True={'HS' if row['true']==1 else 'NHS'}, Pred={'HS' if row['pred']==1 else 'NHS'}")
    
    print("\n" + "="*60)
    print("TOP 5 LOWEST RISK")
    print("="*60)
    for _, row in results.tail(5).iterrows():
        check = "✓" if row['pred'] == row['true'] else "✗"
        print(f"{check} {row['name']}: P={row['prob']:.3f} | True={'HS' if row['true']==1 else 'NHS'}, Pred={'HS' if row['pred']==1 else 'NHS'}")
    
    # Simple plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    cm = np.array([[tn, fp], [fn, tp]])
    axes[0].imshow(cm, cmap='Blues')
    axes[0].set_title('Confusion Matrix')
    for i in range(2): 
        for j in range(2): 
            axes[0].text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=20, fontweight='bold')
    axes[0].set_xticks([0,1])
    axes[0].set_yticks([0,1])
    axes[0].set_xticklabels(['NHS', 'HS'])
    axes[0].set_yticklabels(['NHS', 'HS'])
    
    axes[1].hist(probs, bins=15, color='steelblue', edgecolor='black')
    axes[1].axvline(threshold, color='red', linestyle='--', linewidth=2)
    axes[1].set_title('Probability Distribution')
    axes[1].set_xlabel('Probability')
    
    axes[2].bar(['Accuracy', 'Precision', 'Recall', 'F1'], [acc, prec, rec, f1], color=['green', 'orange', 'blue', 'purple'])
    axes[2].set_ylim([0, 1])
    axes[2].set_title('Metrics')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150)
    print(f"\n✓ Visualization saved: test_results.png")
    print("\n" + "="*60)


if __name__ == "__main__":
    test_model('iccad_hotspot_hr.pkl', n_samples=20)