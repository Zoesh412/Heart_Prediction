import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
import matplotlib.pyplot as plt
import time
import wfdb
import random

# 设置随机种子
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

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

# ============ 4. 构建SimpleRNN模型 ============
def create_rnn_single(input_shape):
    """单步预测SimpleRNN"""
    model = Sequential([
        SimpleRNN(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_rnn_multi(input_shape, output_size=1500):
    """多步预测SimpleRNN"""
    model = Sequential([
        SimpleRNN(64, activation='tanh', input_shape=input_shape),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dense(output_size)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ============ 5. 训练函数 ============
def train_model(model, X_train, y_train, X_test, y_test, epochs, model_name):
    """训练模型并评估"""
    print(f"\n{'='*50}")
    print(f"训练 {model_name}")
    print(f"{'='*50}")
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=0
    )
    
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=1,
        batch_size=32,
        callbacks=[early_stopping]
    )
    train_time = time.time() - start_time
    
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    
    start_time = time.time()
    for _ in range(10):
        _ = model.predict(X_test[:1], verbose=0)
    inference_time_10 = time.time() - start_time
    
    print(f"\n结果汇总：")
    print(f"  MAE: {mae:.4f}秒")
    print(f"  10次推理总时间：{inference_time_10:.2f}秒")
    print(f"  训练时间：{train_time:.2f}秒")
    print(f"  实际训练轮数: {len(history.history['loss'])}")
    
    return {
        'model_name': model_name,
        'mae': mae,
        'inference_time_10': inference_time_10,
        'train_time': train_time,
        'history': history,
        'model': model
    }

# ============ 6. PPT风格的单步预测图 ============
def plot_ppt_style_single(model, X_test, y_test, save_name='RNN_单步预测结果.png'):
    """
    画单步预测的实际值 vs 预测值对比图
    """
    print("\n" + "="*50)
    print("生成SimpleRNN单步预测对比图")
    print("="*50)
    
    # 取前100个测试样本做预测
    n_samples = min(100, len(X_test))
    y_true_all = []
    y_pred_all = []
    
    for i in range(n_samples):
        X_sample = X_test[i:i+1]
        y_true = y_test[i]
        y_pred = model.predict(X_sample, verbose=0)[0, 0]
        
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
    
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    
    # 画图
    plt.figure(figsize=(14, 6))
    
    x_axis = np.arange(n_samples)
    
    plt.plot(x_axis, y_true_all, 'b-', label='Actual', linewidth=1.5, alpha=0.8)
    plt.plot(x_axis, y_pred_all, 'r--', label='Predicted (SimpleRNN)', linewidth=1.5, alpha=0.8)
    
    plt.title('SimpleRNN Single-Step Prediction (20→1)')
    plt.xlabel('Sample Number')
    plt.ylabel('Heartbeat Interval (Seconds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"SimpleRNN单步预测图已保存为：{save_name}")
    
    # 计算并打印MAE
    mae = np.mean(np.abs(y_pred_all - y_true_all))
    print(f"SimpleRNN单步预测MAE: {mae:.4f}秒")
    
    return y_true_all, y_pred_all

# ============ 7. PPT风格的多步预测对比图 ============
def plot_ppt_style_multistep(model, X_test, y_test, train_mean, train_std, save_name='RNN_多步预测结果.png'):
    """
    画多步预测的实际值 vs 预测值对比图
    """
    print("\n" + "="*50)
    print("生成SimpleRNN多步预测对比图")
    print("="*50)
    
    # 取第一个测试样本做预测
    X_sample = X_test[:1]
    y_true = y_test[0] * train_std + train_mean  # 还原原始尺度
    
    # 预测
    y_pred_norm = model.predict(X_sample, verbose=0)[0]
    y_pred = y_pred_norm * train_std + train_mean  # 还原原始尺度
    
    # 画图
    plt.figure(figsize=(14, 6))
    
    # 取前500个点看得清楚些
    steps = min(500, len(y_true))
    x_axis = np.arange(steps)
    
    plt.plot(x_axis, y_true[:steps], 'b-', label='Actual', linewidth=1.5, alpha=0.8)
    plt.plot(x_axis, y_pred[:steps], 'r--', label='Predicted (SimpleRNN)', linewidth=1.5, alpha=0.8)
    
    plt.title('SimpleRNN Multi-Step Prediction (1500→1500)')
    plt.xlabel('Beat Number')
    plt.ylabel('Time Interval (Seconds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"SimpleRNN多步预测图已保存为：{save_name}")
    
    # 计算并打印MAE
    mae = np.mean(np.abs(y_pred - y_true))
    print(f"该样本的MAE: {mae:.4f}秒")
    
    return y_true, y_pred

# ============ 主程序 ============
def main():
    folder_path = "files"
    
    print("="*60)
    print("开始SimpleRNN模型测试")
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
    print("SimpleRNN单步预测 (输入20个，输出1个)")
    print("="*60)
    
    X_train_single, y_train_single = create_single_step_sequences(train_data, 20, 1)
    X_test_single, y_test_single = create_single_step_sequences(test_data, 20, 1)
    
    X_train_single_3d = X_train_single.reshape(-1, 20, 1)
    X_test_single_3d = X_test_single.reshape(-1, 20, 1)
    
    print(f"训练样本数：{len(X_train_single)}")
    print(f"测试样本数：{len(X_test_single)}")
    
    model_single = create_rnn_single((20, 1))
    single_result = train_model(
        model_single, 
        X_train_single_3d, y_train_single,
        X_test_single_3d, y_test_single, 
        epochs=20,
        model_name="SimpleRNN单步"
    )
    
    # ===== 生成PPT风格的单步预测图 =====
    plot_ppt_style_single(
        model_single,
        X_test_single_3d,
        y_test_single,
        save_name='SimpleRNN_单步预测结果.png'
    )
    
    # ===== 多步预测 =====
    print("\n" + "="*60)
    print("SimpleRNN多步预测 (输入1500个，输出1500个)")
    print("="*60)
    
    X_train_multi, y_train_multi = create_multistep_sequences(train_data, 1500, 1500)
    X_test_multi, y_test_multi = create_multistep_sequences(test_data, 1500, 1500)
    
    if len(X_train_multi) > 0 and len(X_test_multi) > 0:
        # 重塑数据
        X_train_multi_3d = X_train_multi.reshape(-1, 1500, 1)
        X_test_multi_3d = X_test_multi.reshape(-1, 1500, 1)
        
        # 数据标准化
        train_mean = np.mean(X_train_multi_3d)
        train_std = np.std(X_train_multi_3d)
        X_train_multi_norm = (X_train_multi_3d - train_mean) / train_std
        X_test_multi_norm = (X_test_multi_3d - train_mean) / train_std
        y_train_multi_norm = (y_train_multi - train_mean) / train_std
        y_test_multi_norm = (y_test_multi - train_mean) / train_std
        
        # 创建多步预测模型
        model_multi = create_rnn_multi((1500, 1), 1500)
        model_multi.summary()
        
        # 训练
        multi_result = train_model(
            model_multi,
            X_train_multi_norm, y_train_multi_norm,
            X_test_multi_norm, y_test_multi_norm,
            epochs=30,
            model_name="SimpleRNN多步"
        )
        
        # 转换MAE回原始尺度
        if multi_result:
            multi_result['mae'] = multi_result['mae'] * train_std
        
        # ===== 生成PPT风格的多步预测对比图 =====
        if multi_result:
            plot_ppt_style_multistep(
                model_multi,
                X_test_multi_norm,
                y_test_multi_norm,
                train_mean,
                train_std,
                save_name='SimpleRNN_多步预测结果.png'
            )
    
    print("\n" + "="*60)
    print("SimpleRNN模型测试完成！")
    print("生成的文件：")
    print("  1. SimpleRNN_单步预测结果.png - 单步预测对比图")
    print("  2. SimpleRNN_多步预测结果.png - 多步预测对比图")
    print("="*60)

if __name__ == "__main__":
    main()
