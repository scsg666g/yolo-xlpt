"""Refresh the standalone Agent's portable dataset code without copying configuration."""
from pathlib import Path


def main():
    agent_dir = Path(__file__).resolve().parent
    backend = agent_dir.parent / 'backend' / 'app'
    target = agent_dir / 'monitor_agent.py'
    script = target.read_text(encoding='utf-8')
    for variable, module in [('DATASET_FILES', 'dataset_files'), ('DATASET_DISCOVERY', 'dataset_discovery_files')]:
        start = script.index(variable + '_SOURCE = ')
        end = script.index('\n' + variable + ' =', start)
        script = script[:start] + variable + '_SOURCE = ' + repr((backend / (module + '.py')).read_text(encoding='utf-8')) + script[end:]
    compile(script, str(target), 'exec')
    target.write_text(script, encoding='utf-8')


if __name__ == '__main__':
    main()
