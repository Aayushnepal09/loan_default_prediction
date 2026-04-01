#In order to sanve time and computational power we have added a feature to skip the model training step if the model is already trained. 
#this was addaded later on so i have added a comment here to let you know about it.


import os
import subprocess
import sys

def run_script(script_path):
    print(f"\n{'='*50}")
    print(f"Running {script_path}...")
    print(f"{'='*50}")
    
    # Run the script using the current Python interpreter
    result = subprocess.run([sys.executable, script_path], check=False)
    
    if result.returncode != 0:
        print(f"\n❌ Error running {script_path}. Exiting pipeline.")
        sys.exit(result.returncode)
    else:
        print(f"\n✅ {script_path} completed successfully.")

def main():
    # Identify the project root and relevant directories
    root_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(root_dir, "src")
    models_script_dir = os.path.join(src_dir, "models")
    models_dir = os.path.join(root_dir, "models")
    
    # 1. Data Processing Scripts (Always run these in order)
    data_scripts = [
        "01_data_loading.py",
        "02_data_inspection.py",
        "03_data_cleaning.py",
        "04_data_splitting.py",
        "05_data_eda.py",
        "06_data_processing_pipeline.py"
    ]
    
    print("🚀 Starting Main Pipeline Execution...")
    
    for script in data_scripts:
        script_path = os.path.join(src_dir, script)
        if os.path.exists(script_path):
            run_script(script_path)
        else:
            print(f"\n⚠️ Warning: {script_path} not found, skipping...")

    # 2. Conditionally skip Model Training (13_model_selection.py)
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    model_selection_script = os.path.join(models_script_dir, "13_model_selection.py")
    
    if os.path.exists(best_model_path):
        print(f"\n{'='*50}")
        print(f"🟢 Found existing model at {best_model_path}.")
        print(f"⏩ Skipping intensive model training ({model_selection_script}).")
        print(f"   (Delete the model file if you wish to force re-training)")
        print(f"{'='*50}")
    else:
        if os.path.exists(model_selection_script):
            run_script(model_selection_script)
        else:
            print(f"\n⚠️ Warning: {model_selection_script} not found!")

    # 3. Final Evaluation
    final_eval_script = os.path.join(models_script_dir, "14_final_evaluation.py")
    if os.path.exists(final_eval_script):
        run_script(final_eval_script)
    else:
        print(f"\n⚠️ Warning: {final_eval_script} not found.")

    print("\n🎉 Pipeline execution beautifully finished!")

if __name__ == "__main__":
    main()
