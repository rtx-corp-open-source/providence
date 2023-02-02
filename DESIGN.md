<!-- 
**Raytheon Technologies proprietary**
Export controlled - see license file
-->
# A coarse API by design
Providence was built with the end user in-mind, with a layered designed focused on providing appropriately coarse APIs.
Too often library designers emphasize providing an atomic scalpal or a crane construction company, when researchers just want to build a shed. Worse yet, they want to build a car and only have tools for building a shed!

With `providence`, we seek to strike the balance:
- if you want to leverage high-level APIs and just run some quick tests, we [make that available](./examples/01_2_high_level_api_with_overrides.py)
- if you want to pull things apart and start experimenting with custom [loss functions](./examples/05_custom_loss_with_low_level_api.py) or [architectures](./examples/02_mid_level_api_custom_model.py), there are ample provisions for that.
- custom training loops? We made room for that, [too](./examples/03_custom_training_via_high_and_mid_level_api.py)!

For more on this mentality, consider watching this videos by two of the greatest software engineers in the industry:
- first, on API granularity by Casey Muratori: https://www.youtube.com/watch?v=ZQ5_u8Lgvyk
- second, on layered APIs by Jeremy Howard: https://www.youtube.com/watch?v=imsBjsWLJzE

And remember: if something isn't implemented in the core library, it might just be an [example](./examples/) that shows how easy Providence is extended. You still have to write _some_ code 😉


## API conventions
Analogous to [the Python core library][itertools-cycle-example] (where classes feels like functions), we have many constructor functions which are defined to look like classes.
There is a meaningful class underneath, but implementations are usually straightforward and therefore left with a terse or non-existent docstring. One such example is `ProvidenceLSTM`:

```python
def ProvidenceLSTM(
    input_size: int,
    hidden_size: int = 24,
    num_layers: int = 2,
    dropout: float = 0.,
    *,
    activation: str = 'weibull',
    device: device = 'cpu'
) -> ProvidenceRNN:
    lstm = LSTM(input_size, hidden_size, num_layers, dropout=dropout)
    return ProvidenceRNN(lstm, activation=activation, device=device)
```
Both `ProvidenceRNN` and `torch.nn.LSTM` are heavily documented; this function would have little to add to the conversation.

Another point of interest is our encouragement of shadowing via inheritance with `providence.distributions.SurvivalAnalysisDistribution`.
This is convention that affords extensibility of the framework, at the cost of potentially being frown upon by Guiddo.

Lastly, we don't write _everything_ for you. We focused on providing useful functions and classes that can be composed to effective research and productization ends.
But we didn't go out of our way to write a `Trainer` a la PyTorch Lightning or force everything to use a `.fit()`. That was not the goal and so we did not do it.

## Documentation conventions
In general, rather than "saying" the function definition in plain text, the documentation is oriented towards either
1. explaining why something exists
2. with what else you might use it.

That is to say, of the four fundamental categories of documenation software developers prefer, we prioritize them in this order
1. Knowledge base / storytelling
1. tutorial / walkthrough
1. API reference
1. Troubleshooting

It may be strange to see "Troubleshooting" at the bottom of the list, but if you make it through the first three - and we did our job right - you shouldn't have any `providence`-related troubles. It will just be PyTorch or (more generally) Python.
Python text-editors and notebook environments do a surprisingly good job of manifesting type information. We assume these tools are at your disposal, making explanatory source code (rather than prose examples) much easier to surface.

In that spirit
- If a function is only one or two effective (i.e. on a single line before formatting) source-lines of code (SLOC), we don't write a docstring _unless_ the usage is particularly opaque.
- Every Python file has a header that explains the mentality that went into that modules creation.
- Most functions have a demonstrative code example, and more complex (or obscure in motivation) functions have more code examples.
  - If something is both heavy on the implementation, and sparse on documentation, it is probably a nested function that is explained by the context and has a [good name](https://stackoverflow.com/questions/1991324/function-naming-conventions).



<!-- links -->
[itertools-cycle-example]: https://docs.python.org/3/library/itertools.html?highlight=cycle#itertools.cycle