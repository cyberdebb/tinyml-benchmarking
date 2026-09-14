import os
from pathlib import Path
import subprocess
import sys

def run_all_classifiers():
	classifiers = (
		'cnn_classifier.py',
		'mlp_classifier.py',
		'random_forest_classifier.py',
		'svm_classifier.py',
	)
 
	classifiers_directory = Path(__file__).resolve().parent / 'classifiers'
	project_directory = Path(__file__).resolve().parents[1]
	environment = os.environ.copy()
	environment['PYTHONPATH'] = str(Path(__file__).resolve().parent)

	print('[START] Running all classifiers sequentially.')
	failed_classifiers = []

	for classifier in classifiers:
		classifier_path = classifiers_directory / classifier
		print(f'\n[RUN] Starting {classifier}...')
		result = subprocess.run(
			[sys.executable, str(classifier_path)],
			cwd=project_directory,
			env=environment,
			check=False,
		)

		if result.returncode == 0:
			print(f'[DONE] {classifier} completed successfully.')
		else:
			print(f'[ERROR] {classifier} failed with exit code {result.returncode}.')
			failed_classifiers.append(classifier)

	if failed_classifiers:
		print(f'[SUMMARY] Failed classifiers: {", ".join(failed_classifiers)}')
		raise SystemExit(1)

	print('[SUMMARY] All classifiers completed successfully.')


if __name__ == '__main__':
	run_all_classifiers()
