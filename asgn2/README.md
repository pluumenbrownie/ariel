To run a simulation:
```
$ python asgn2/cli.py run --help
Usage: cli.py run [OPTIONS]

Options:
  -t, --type [adaptive|uniform|nobrain]
                                  The Brain class used.
  -p, --pop-size INTEGER          The amount of individuals tested each
                                  generation.
  -m, --max-gens INTEGER          The maximum number of generations to run.
  -d, --save-dir TEXT             Where to save any generated files.
  -f, --postfix TEXT              Postfix used on any saved files.
  --help                          Show this message and exit.
$ python asgn2/cli.py run -t adaptive -p 100 -m 2000 -f selfadaptive_4
0%|                                                 | 0/2000 [00:00<?, ?it/s]   0%|                                                 | 0/2000 [00:02<?, ?it/s]
```

To show a controller in action in a MuJoCo window:
```
$ python asgn2/cli.py show --help
Usage: cli.py show [OPTIONS] BRAIN_FILE

Options:
  --help  Show this message and exit.
$ python asgn2/cli.py show asgn2/results/SelfAdaptiveBrain_selfadaptive_3.json
```
