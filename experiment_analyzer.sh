for session_path in $(find session -name server); do
  env session_path="${session_path}" python3 experiment_analyzer.py
done
