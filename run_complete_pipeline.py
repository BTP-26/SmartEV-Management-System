import os
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path

project_root = Path(__file__).parent
sys.path.append(str(project_root))

def check_data_availability():
    print("checking data availability...")
    
    from shared.config import get_config
    config = get_config()
    paths = config.get_paths_config()
    
    braking_files = [
        os.path.join(paths['data']['braking'], 'X_train_real.npy'),
        os.path.join(paths['data']['braking'], 'y_class_train_real.npy'), 
        os.path.join(paths['data']['braking'], 'X_val_real.npy'),
        os.path.join(paths['data']['braking'], 'y_class_val_real.npy')
    ]
    
    soc_files = [
        os.path.join(paths['data']['soc'], 'X_train_real.npy'),
        os.path.join(paths['data']['soc'], 'y_train_real.npy'),
        os.path.join(paths['data']['soc'], 'X_val_real.npy'), 
        os.path.join(paths['data']['soc'], 'y_val_real.npy')
    ]
    
    missing_braking = [f for f in braking_files if not os.path.exists(f)]
    missing_soc = [f for f in soc_files if not os.path.exists(f)]
    
    if missing_braking:
        print(f"missing braking data: {missing_braking}")
        return False
    
    if missing_soc:
        print(f"missing soc data: {missing_soc}")
        return False
    
    print("all datasets available")
    return True

def generate_braking_data():
    print("generating braking dataset...")
    
    try:
        from modules.braking.data.preprocess_real_data import process_all_data
        process_all_data()
        print("braking dataset generated")
        return True
        
    except Exception as e:
        print(f"error: {e}")
        return False

def generate_soc_data():
    print("generating soc dataset...")
    
    try:
        from modules.soc.data.preprocess_real_data import process_all_data
        process_all_data()
        print("soc dataset generated")
        return True
        
    except Exception as e:
        print(f"error: {e}")
        return False

def check_model_availability():
    print("checking model availability...")
    
    models = {
        "braking": "modules/braking/models/final_multitask_model.pth",
        "soc": "modules/soc/models/lstm_cnn_attention_soc.pth"
    }
    
    missing = []
    for name, path in models.items():
        if not os.path.exists(path):
            missing.append(f"{name}: {path}")
    
    if missing:
        print(f"missing models: {missing}")
        return False
    
    print("all models available")
    return True

def train_models():
    print("training models...")
    
    try:
        print("training braking model...")
        os.system(f"cd {project_root} && source .venv/bin/activate && python modules/train/train_braking.py")
        
        # train soc model
        print("training soc model...")
        os.system(f"cd {project_root} && source .venv/bin/activate && python modules/train/train_soc.py")
        
        print("model training completed")
        return True
        
    except Exception as e:
        print(f"error training models: {e}")
        return False

def calculate_and_display_model_metrics():
    """calculate and display model quality metrics by evaluating existing models"""
    print("calculating model quality metrics...")
    
    metrics = {}
    
    try:
        import torch
        import torch.nn as nn
        from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
        from shared.config import get_config
        from shared.dataset_loader import get_dataset_loader
        from modules.braking.models.multitask_lstm_cnn_attention import MultitaskLSTMCNNAttention
        from modules.soc.models.lstm_cnn_attention_soc import LSTMCNNAttentionSoC
        
        config = get_config()
        dataset_loader = get_dataset_loader()
        device = torch.device("cpu")
        
        metrics = {}
        
        # evaluate braking model
        print("evaluating braking model...")
        try:
            (X_train, X_val, X_test, 
             y_int_train, y_int_val, y_int_test,
             y_class_train, y_class_val, y_class_test) = dataset_loader.load_braking_dataset()
            
            model = MultitaskLSTMCNNAttention(
                input_dim=7,
                cnn_channels=32,
                lstm_hidden=64,
                num_lstm_layers=1,
                dropout_rate=0.0
            )
            
            model_path = "modules/braking/models/final_multitask_model.pth"
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            

            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_class_test_tensor = torch.tensor(y_class_test, dtype=torch.long).to(device)
            
            with torch.no_grad():
                cls_outputs, _ = model(X_test_tensor)
                _, predicted = torch.max(cls_outputs.data, 1)
                
                y_true = y_class_test_tensor.cpu().numpy()
                y_pred = predicted.cpu().numpy()
                
                # calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                f1_macro = f1_score(y_true, y_pred, average='macro')
                f1_weighted = f1_score(y_true, y_pred, average='weighted')
                
                metrics['braking'] = {
                    'test_accuracy': float(accuracy),
                    'test_f1_macro': float(f1_macro),
                    'test_f1_weighted': float(f1_weighted),
                    'test_samples': len(y_true)
                }
                
        except Exception as e:
            print(f"  error evaluating braking model: {e}")
        
        # evaluate soc model
        print("evaluating soc model...")
        try:

            X_test = np.load("modules/soc/data/X_test_real.npy")
            y_test = np.load("modules/soc/data/y_test_real.npy")
            
            print(f"  original X_test shape: {X_test.shape}")
            print(f"  original y_test shape: {y_test.shape}")
            
            # smaller subset for evaluation to avoid memory issues
            max_samples = min(1000, len(X_test))  
            X_test_subset = X_test[:max_samples]
            y_test_subset = y_test[:max_samples]
            
            print(f"  using {len(X_test_subset)} samples for evaluation (subset of {len(X_test)})")
            print(f"  test data shape: {X_test_subset.shape}")
            
            model = LSTMCNNAttentionSoC(
                input_dim=3,
                cnn_channels=64,
                lstm_hidden=128,
                num_lstm_layers=2,
                dropout=0.2
            )
            
            model_path = "modules/soc/models/lstm_cnn_attention_soc.pth"
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            
            batch_size = 32
            all_predictions = []
            
            with torch.no_grad():
                for i in range(0, len(X_test_subset), batch_size):
                    batch_x = torch.tensor(X_test_subset[i:i+batch_size], dtype=torch.float32).to(device)
                    batch_outputs = model(batch_x)
                    all_predictions.extend(batch_outputs.cpu().numpy())
                    
                    # clear memory
                    del batch_x, batch_outputs
                    if i % (batch_size * 10) == 0:  # periodic cleanup
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            y_pred = np.array(all_predictions)
            
            # calculate regression metrics
            mse = mean_squared_error(y_test_subset, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_subset, y_pred)
            mape = np.mean(np.abs((y_test_subset - y_pred) / (y_test_subset + 1e-8))) * 100
            
            metrics['soc'] = {
                'test_rmse': float(rmse),
                'test_mae': float(mae),
                'test_mape': float(mape),
                'test_samples': len(y_test_subset)
            }
            
            print(f"  soc evaluation completed successfully")
                
        except Exception as e:
            print(f"  error evaluating soc model: {e}")
            import traceback
            traceback.print_exc()
        
        # display metrics
        print("model quality metrics")
        if 'braking' in metrics:
            print("braking model performance:")
            b_metrics = metrics['braking']
            print(f"  test accuracy: {b_metrics['test_accuracy']:.4f}")
            print(f"  test f1-score (macro): {b_metrics['test_f1_macro']:.4f}")
            print(f"  test f1-score (weighted): {b_metrics['test_f1_weighted']:.4f}")
            print(f"  test samples: {b_metrics['test_samples']}")
        else:
            print("\nbraking model: evaluation failed")
        
        if 'soc' in metrics:
            print("\nsoc model performance:")
            s_metrics = metrics['soc']
            print(f"  test rmse: {s_metrics['test_rmse']:.4f}")
            print(f"  test mae: {s_metrics['test_mae']:.4f}")
            print(f"  test mape: {s_metrics['test_mape']:.2f}%")
            print(f"  test samples: {s_metrics['test_samples']}")
        else:
            print("\nsoc model: evaluation failed")
        
        return metrics
        
    except Exception as e:
        print(f"error calculating model metrics: {e}")
        return {}

def test_enhanced_pipeline():
    print("testing enhanced pipeline...")
    
    try:
        from shared.enhanced_utils import EnhancedEVPipeline
        
        print("initializing pipeline...")
        pipeline = EnhancedEVPipeline()
        
        print("generating test data...")
        driving_window, battery_window = pipeline.generate_sample_inputs()
        current_soc = 0.45
        
        # test basic inference
        print("testing basic inference...")
        start_time = time.time()
        basic_result = pipeline.run_single(driving_window, battery_window, current_soc)
        basic_time = time.time() - start_time
        
        print(f"basic inference: {basic_result['system_action']}")
        print(f"inference time: {basic_time*1000:.2f}ms")
        
        # test cognitive inference
        print("testing cognitive inference...")
        start_time = time.time()
        cognitive_result = pipeline.run_with_cognitive(
            driving_window, battery_window, current_soc, driver_id="test_driver_001"
        )
        cognitive_time = time.time() - start_time
        
        print(f"driver style: {cognitive_result['cognitive']['driver_profile']['driving_style']}")
        print(f"prediction confidence: {cognitive_result['cognitive']['prediction_confidence']:.2f}")
        print(f"cognitive action: {cognitive_result['system_action']}")
        print(f"inference time: {cognitive_time*1000:.2f}ms")
        
        # test batch inference
        print("testing batch inference...")
        driving_windows, battery_windows = pipeline.generate_sample_inputs(10)
        current_socs = [0.3 + i*0.05 for i in range(10)]
        
        start_time = time.time()
        batch_results = pipeline.run_batch(driving_windows, battery_windows, current_socs)
        batch_time = time.time() - start_time
        
        throughput = len(batch_results) / batch_time
        print(f"batch inference: {len(batch_results)} samples")
        print(f"throughput: {throughput:.1f} samples/second")
        
        # get cognitive summary
        print("getting cognitive summary...")
        summary = pipeline.get_cognitive_summary()
        print(f"active drivers: {summary.get('active_drivers', 0)}")
        print(f"adaptation level: {summary.get('adaptation_level', 0):.2f}")
        
        print("enhanced pipeline test completed")
        return True
        
    except Exception as e:
        print(f"error testing enhanced pipeline: {e}")
        return False

def run_performance_benchmark():
    print("running performance benchmark...")
    
    try:
        from shared.enhanced_utils import EnhancedEVPipeline
        
        pipeline = EnhancedEVPipeline()
        print("benchmarking single inference...")
        driving_window, battery_window = pipeline.generate_sample_inputs()
        current_soc = 0.5
        
        times = []
        for i in range(100):
            start_time = time.time()
            result = pipeline.run_single(driving_window, battery_window, current_soc)
            times.append(time.time() - start_time)
        
        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        
        print(f"average inference time: {avg_time:.2f}±{std_time:.2f}ms")
        print(f"min/max time: {min(times)*1000:.2f}/{max(times)*1000:.2f}ms")
        
        print("benchmarking batch inference...")
        batch_sizes = [1, 10, 50, 100]
        
        for batch_size in batch_sizes:
            print(f"generating {batch_size} samples...")
            driving_windows, battery_windows = pipeline.generate_sample_inputs(batch_size)
            
            if batch_size == 1:
                if not isinstance(driving_windows, list):
                    driving_windows = [driving_windows]
                if not isinstance(battery_windows, list):
                    battery_windows = [battery_windows]
            
            # ensure matching lengths for batch processing
            current_socs = [0.5 + i*0.001 for i in range(batch_size)]  # Smaller increments to stay in [0,1]
            current_socs = [min(0.95, max(0.05, soc)) for soc in current_socs]  # Clamp to valid range
            
            print(f"generated {len(driving_windows)} driving windows, {len(battery_windows)} battery windows, {len(current_socs)} soc values")
            
            if len(driving_windows) != len(battery_windows) or len(driving_windows) != len(current_socs):
                print(f"error: length mismatch - driving: {len(driving_windows)}, battery: {len(battery_windows)}, soc: {len(current_socs)}")
                continue
            
            start_time = time.time()
            results = pipeline.run_batch(driving_windows, battery_windows, current_socs)
            batch_time = time.time() - start_time
            
            throughput = batch_size / batch_time
            print(f"batch size {batch_size}: {throughput:.1f} samples/second")
        
        print("performance benchmark completed")
        return True
        
    except Exception as e:
        print(f"error in performance benchmark: {e}")
        return False

def main():
    print("ev smart management system - pipeline test")
    
    results = {}
    
    if not check_data_availability():
        print("generating missing datasets...")
        results['braking_data'] = generate_braking_data()
        results['soc_data'] = generate_soc_data()
    else:
        results['braking_data'] = True
        results['soc_data'] = True
    
    if not check_model_availability():
        print("training missing models...")
        results['model_training'] = train_models()
    else:
        results['model_training'] = True
    
    if not (results.get('braking_data', False) and results.get('soc_data', False) and results.get('model_training', False)):
        print("skipping pipeline tests - some steps failed")
        return
    
    # calculate and display model quality metrics
    model_metrics = calculate_and_display_model_metrics()
    results['model_metrics'] = model_metrics
    
    print("testing enhanced pipeline...")
    results['enhanced_pipeline'] = test_enhanced_pipeline()
    
    print("performance benchmark...")
    results['performance'] = run_performance_benchmark()
    
    print("pipeline test summary")
    print(f"data generation: {'pass' if results.get('braking_data', False) else 'fail'}")
    print(f"model training: {'pass' if results.get('model_training', False) else 'fail'}")
    print(f"enhanced pipeline: {'pass' if results.get('enhanced_pipeline', False) else 'fail'}")
    print(f"performance: {'pass' if results.get('performance', False) else 'fail'}")
    
    overall = all([results.get(k, False) for k in ['braking_data', 'model_training', 'enhanced_pipeline', 'performance']])
    print(f"\noverall: {'all tests passed' if overall else 'some tests failed'}")
    
    if overall:
        print("ev smart management system is fully operational!")
        # save final report with metrics
        save_final_report_with_metrics(results)
    else:
        print("some tests failed - check logs above")

def save_final_report_with_metrics(results):
    """save final report with model quality metrics"""
    import json
    from datetime import datetime
    
    try:
        # ensure training_checkpoints directory exists
        os.makedirs("training_checkpoints", exist_ok=True)
        
        # create comprehensive report
        report = {
            "timestamp": datetime.now().isoformat(),
            "status": "operational",
            "pipeline_results": {
                "data_generation": results.get('braking_data', False),
                "model_training": results.get('model_training', False),
                "enhanced_pipeline": results.get('enhanced_pipeline', False),
                "performance": results.get('performance', False)
            },
            "model_quality_metrics": results.get('model_metrics', {}),
            "summary": {
                "overall_status": "all tests passed" if all([
                    results.get('braking_data', False),
                    results.get('model_training', False),
                    results.get('enhanced_pipeline', False),
                    results.get('performance', False)
                ]) else "some tests failed",
                "components_successful": sum([
                    results.get('braking_data', False),
                    results.get('model_training', False),
                    results.get('enhanced_pipeline', False),
                    results.get('performance', False)
                ]),
                "total_components": 4
            }
        }
        
        # save report
        report_path = "training_checkpoints/final_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"final report saved: {report_path}")
        
    except Exception as e:
        print(f"error saving final report: {e}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
