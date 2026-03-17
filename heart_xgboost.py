import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
import wfdb
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# 设置随机种子
random.seed(42)
np.random.seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============ 1. 数据读取 ============
def load_rr_from_ecg(data_path):
    """使用wfdb读取.ecg文件，提取RR间隔"""
    all_rr_intervals = []
    file_count = 0
    
    print(f"正在从文件夹读取ECG数据：{data_path}")
    
    if not os.path.exists(data_path):
        print(f"错误：文件夹 {data_path} 不存在！")
        return np.array([])
    
    for file in os.listdir(data_path):
        if file.startswith('chf') and file.endswith('.ecg'):
            profile_name = file.split('.')[0]
            
            try:
                ann_path = os.path.join(data_path, profile_name)
                rec = wfdb.rdann(ann_path, 'ecg')
                rr = np.diff(rec.sample) / rec.fs
                all_rr_intervals.extend(rr)
                file_count += 1
                print(f"  读取 {profile_name}: {len(rr)} 个RR间隔")
            except Exception as e:
                print(f"  跳过 {file}: {e}")
    
    print(f"\n共读取 {file_count} 个文件，{len(all_rr_intervals)} 个RR间隔数据点")
    return np.array(all_rr_intervals)

# ============ 2. 数据预处理 ============
def preprocess_rr_data(rr_data):
    """预处理数据：过滤异常值"""
    print("\n开始数据预处理...")
    original_len = len(rr_data)
    
    # 步骤1：过滤非正常区间 (0.3-2秒)
    mask1 = (rr_data >= 0.3) & (rr_data <= 2.0)
    data = rr_data[mask1]
    print(f"步骤1过滤后：{len(data)} 个数据点 (移除{original_len - len(data)}个异常值)")
    
    # 步骤2：过滤心跳突变（前后差>0.5秒）
    mask2 = np.ones(len(data), dtype=bool)
    for i in range(1, len(data)-1):
        if abs(data[i] - data[i-1]) > 0.5 or abs(data[i+1] - data[i]) > 0.5:
            mask2[i] = False
    
    if len(data) > 1:
        if abs(data[1] - data[0]) > 0.5:
            mask2[0] = False
        if abs(data[-1] - data[-2]) > 0.5:
            mask2[-1] = False
    
    data = data[mask2]
    print(f"步骤2过滤后：{len(data)} 个数据点 (移除突变点)")
    
    # 步骤3：取前30000个干净数据
    target_size = 30000
    if len(data) >= target_size:
        clean_data = data[:target_size]
        print(f"步骤3：取前{target_size}个数据")
    else:
        clean_data = data
        print(f"警告：数据不足{target_size}，只有{len(data)}个")
    
    print(f"\n预处理完成：")
    print(f"  数据范围：{clean_data.min():.3f} - {clean_data.max():.3f}秒")
    print(f"  平均值：{clean_data.mean():.3f}秒")
    print(f"  标准差：{clean_data.std():.3f}秒")
    print(f"  最终数据量：{len(clean_data)}个")
    
    return clean_data

# ============ 3. 创建序列数据 ============
def create_single_step_sequences(data, input_len=20, output_len=1):
    """创建单步预测样本"""
    X, y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len])
    return np.array(X), np.array(y)

def create_multistep_sequences(data, input_len=1500, output_len=1500):
    """创建多步预测样本"""
    X, y = [], []
    total_len = input_len + output_len
    
    # 步长设为200
    stride = 200
    print(f"  生成样本中... 总数据长度: {len(data)}, 步长: {stride}")
    
    for i in range(0, len(data) - total_len, stride):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len:i+total_len])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"  共生成 {len(X)} 个样本")
    print(f"  输入形状: {X.shape}, 输出形状: {y.shape}")
    
    return X, y

# ============ 4. XGBoost单步预测 ============
def train_xgboost_single(X_train, y_train, X_test, y_test):
    """XGBoost单步预测"""
    print("\n" + "="*50)
    print("训练 XGBoost 单步预测")
    print("="*50)
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练模型
    start_time = time.time()
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    # 预测
    start_time = time.time()
    y_pred = model.predict(X_test_scaled)
    inference_time = time.time() - start_time
    
    # 计算MAE
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n结果汇总：")
    print(f"  MAE: {mae:.4f}秒")
    print(f"  训练时间：{train_time:.2f}秒")
    print(f"  推理时间：{inference_time*1000:.2f}毫秒")
    
    return {
        'mae': mae,
        'model': model,
        'scaler': scaler,
        'y_pred': y_pred,
        'y_test': y_test
    }

# ============ 5. XGBoost多步预测（简化版，只预测前100步） ============
def train_xgboost_multi_simple(X_train, y_train, X_test, y_test, n_outputs=100):
    """XGBoost多步预测 - 只预测前100步"""
    print("\n" + "="*50)
    print("训练 XGBoost 多步预测（前100步）")
    print("="*50)
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 只取前n_outputs个输出
    y_train_subset = y_train[:, :n_outputs]
    
    start_time = time.time()
    models = []
    
    print(f"  训练 {n_outputs} 个XGBoost模型...")
    
    for i in range(n_outputs):
        if i % 20 == 0:
            print(f"    训练第 {i}/{n_outputs} 个模型")
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            n_jobs=1,
            verbosity=0
        )
        model.fit(X_train_scaled, y_train_subset[:, i])
        models.append(model)
    
    train_time = time.time() - start_time
    
    # 预测
    start_time = time.time()
    y_pred = np.zeros((X_test_scaled.shape[0], n_outputs))
    for i, model in enumerate(models):
        y_pred[:, i] = model.predict(X_test_scaled)
    
    inference_time = time.time() - start_time
    
    # 计算MAE
    y_test_subset = y_test[:, :n_outputs]
    mae = np.mean(np.abs(y_pred - y_test_subset))
    
    print(f"\n结果汇总：")
    print(f"  MAE: {mae:.4f}秒")
    print(f"  训练时间：{train_time:.2f}秒")
    print(f"  推理时间：{inference_time*1000:.2f}毫秒")
    
    return {
        'mae': mae,
        'models': models,
        'scaler': scaler,
        'y_pred': y_pred,
        'y_test': y_test_subset
    }

# ============ 6. 画单步预测图（PPT风格） ============
def plot_xgboost_single(result, save_name='XGBoost_单步预测结果.png'):
    """
    画XGBoost单步预测的PPT风格图
    蓝线实际值，红线预测值
    """
    print("\n" + "="*50)
    print("生成XGBoost单步预测PPT风格图")
    print("="*50)
    
    # 取前100个样本
    n_samples = min(100, len(result['y_pred']))
    x_axis = np.arange(n_samples)
    
    plt.figure(figsize=(14, 6))
    
    plt.plot(x_axis, result['y_test'][:n_samples], 'b-', label='Actual', linewidth=1.5, alpha=0.8)
    plt.plot(x_axis, result['y_pred'][:n_samples], 'r--', label='Predicted (XGBoost)', linewidth=1.5, alpha=0.8)
    
    plt.title(f'XGBoost Single-Step Prediction (20→1) - MAE: {result["mae"]:.4f}s')
    plt.xlabel('Sample Number')
    plt.ylabel('Heartbeat Interval (Seconds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"XGBoost单步预测图已保存为：{save_name}")

# ============ 7. 画多步预测图（PPT风格） ============
def plot_xgboost_multi(result, sample_idx=0, save_name='XGBoost_多步预测结果.png'):
    """
    画XGBoost多步预测的PPT风格图
    蓝线实际值，红线预测值
    """
    print("\n" + "="*50)
    print("生成XGBoost多步预测PPT风格图")
    print("="*50)
    
    # 取指定样本
    y_true = result['y_test'][sample_idx]
    y_pred = result['y_pred'][sample_idx]
    
    # 取前500个点
    steps = min(500, len(y_true))
    x_axis = np.arange(steps)
    
    plt.figure(figsize=(14, 6))
    
    plt.plot(x_axis, y_true[:steps], 'b-', label='Actual', linewidth=1.5, alpha=0.8)
    plt.plot(x_axis, y_pred[:steps], 'r--', label='Predicted (XGBoost)', linewidth=1.5, alpha=0.8)
    
    # 计算该样本的MAE
    sample_mae = np.mean(np.abs(y_pred - y_true))
    
    plt.title(f'XGBoost Multi-Step Prediction (1500→1500) - Sample MAE: {sample_mae:.4f}s')
    plt.xlabel('Beat Number')
    plt.ylabel('Time Interval (Seconds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"XGBoost多步预测图已保存为：{save_name}")
    
    return sample_mae

# ============ 主程序 ============
def main():
    folder_path = "files"
    
    print("="*60)
    print("开始XGBoost模型画图（仅PPT风格图）")
    print("="*60)
    
    # 读取数据
    raw_data = load_rr_from_ecg(folder_path)
    if len(raw_data) == 0:
        print("错误：没有读取到任何数据！")
        return
    
    # 预处理
    clean_data = preprocess_rr_data(raw_data)
    if len(clean_data) < 10000:
        print("错误：预处理后数据太少")
        return
    
    # 80/20划分
    train_size = int(0.8 * len(clean_data))
    train_data = clean_data[:train_size]
    test_data = clean_data[train_size:]
    print(f"\n数据划分：")
    print(f"  训练集：{len(train_data)}个 ({len(train_data)/len(clean_data)*100:.1f}%)")
    print(f"  测试集：{len(test_data)}个 ({len(test_data)/len(clean_data)*100:.1f}%)")
    
    # ===== 单步预测 =====
    print("\n" + "="*60)
    print("准备单步预测数据 (20->1)")
    print("="*60)
    
    X_train_single, y_train_single = create_single_step_sequences(train_data, 20, 1)
    X_test_single, y_test_single = create_single_step_sequences(test_data, 20, 1)
    
    print(f"训练样本数：{len(X_train_single)}")
    print(f"测试样本数：{len(X_test_single)}")
    
    # XGBoost单步预测
    single_result = train_xgboost_single(X_train_single, y_train_single, X_test_single, y_test_single)
    
    # 画单步预测图
    plot_xgboost_single(single_result, 'XGBoost_单步预测结果.png')
    
    # ===== 多步预测 =====
    print("\n" + "="*60)
    print("准备多步预测数据 (1500->1500)")
    print("="*60)
    
    X_train_multi, y_train_multi = create_multistep_sequences(train_data, 1500, 1500)
    X_test_multi, y_test_multi = create_multistep_sequences(test_data, 1500, 1500)
    
    if len(X_train_multi) > 0 and len(X_test_multi) > 0:
        # XGBoost多步预测（只预测前100步，节省时间）
        multi_result = train_xgboost_multi_simple(X_train_multi, y_train_multi, X_test_multi, y_test_multi, n_outputs=100)
        
        # 画多步预测图
        plot_xgboost_multi(multi_result, sample_idx=0, save_name='XGBoost_多步预测结果.png')
    
    # ===== 结果汇总 =====
    print("\n" + "="*60)
    print("XGBoost画图完成！")
    print("="*60)
    
    print(f"\n单步XGBoost - MAE: {single_result['mae']:.4f}秒")
    if 'multi_result' in locals():
        print(f"多步XGBoost - MAE: {multi_result['mae']:.4f}秒（前100步）")
    
    print("\n生成的文件：")
    print("  1. XGBoost_单步预测结果.png - PPT风格单步预测图")
    print("  2. XGBoost_多步预测结果.png - PPT风格多步预测图")
    print("="*60)

if __name__ == "__main__":
    main()
