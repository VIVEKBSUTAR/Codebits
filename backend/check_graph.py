from causal_engine.causal_graph import get_causal_graph
print('Initializing graph...')
graph = get_causal_graph('test_zone')
print('Running inference with no evidence...')
res = graph.run_inference()
print(res)
