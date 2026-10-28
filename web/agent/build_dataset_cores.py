"""Refresh the standalone Agent's portable cores without copying configuration."""
import ast
from pathlib import Path


def embed_sources(script, backend):
    """Only edit module statements, never text inside an embedded launcher."""
    for variable, module in [('DATASET_FILES', 'dataset_files'), ('DATASET_DISCOVERY', 'dataset_discovery_files'),
                             ('DATASET_PROVENANCE', 'dataset_provenance'), ('METRIC_FILES', 'metric_files')]:
        statements = ast.parse(script).body
        source = repr((backend / (module + '.py')).read_text(encoding='utf-8'))
        assignment = next((node for node in statements if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == variable + '_SOURCE'
                                   for target in node.targets)), None)
        lines = script.splitlines(keepends=True)
        if assignment is None:
            block = '\n' + variable + '_SOURCE = ' + source + '\n'
            block += variable + ' = {}\nexec(compile(' + variable + '_SOURCE, "<' + module + '>", "exec"), ' + variable + ')\n'
            guard = next(node for node in statements if isinstance(node, ast.If)
                         and isinstance(node.test, ast.Compare)
                         and isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__'
                         and any(isinstance(value, ast.Constant) and value.value == '__main__'
                                 for value in node.test.comparators))
            lines.insert(guard.lineno - 1, block + '\n')
            script = ''.join(lines)
            continue
        lines[assignment.lineno - 1:assignment.end_lineno] = [variable + '_SOURCE = ' + source + '\n']
        script = ''.join(lines)
    compile(script, '<standalone-agent>', 'exec')
    return script


def main():
    agent_dir = Path(__file__).resolve().parent
    backend = agent_dir.parent / 'backend' / 'app'
    target = agent_dir / 'monitor_agent.py'
    script = target.read_text(encoding='utf-8')
    updated = embed_sources(script, backend)
    if updated != script:
        target.write_text(updated, encoding='utf-8')


if __name__ == '__main__':
    main()
