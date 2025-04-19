lowkey kinda gave up at first, but unexpectedly entered the finals
VISION was to make a backtest library for quant trading as beginner friendly as possible
can be kind of braindead to just run the library
but of course, if you have domain knowledge, it would yield better results

addressing judges feedback:
1. parameter optimization
- AUTOMATE and find best n_state based on a range using AIC/BIC
- AIC/BIC(Akaike Information Criterion and Bayesian Information Criterion)
-- WHAT:
	- metric based on likelihood of a model and penalizes complexity
	- ensure statistical model does not overfit
-- WHY:
	- helps to balance simplicity and suitability
	- data's hidden n_states is usually unknown, using too much n_states may capture noise, causing model overfit
-- HOW:
	- lower AIC/BIC, better model
	- BIC is stricter, penalizes complexity more

2. solid model backed by real statistic and theory
- explain why i choose specific features, should link to hypotheses and theory
- Basic feature selection, using Correlation Matrix: (statistical approach)
-- WHAT:
	- square table(means same rows and columns) that compares similarity of each features, called coefficients
	- values:
		- +1 = moves together, perfectly related
		- 0 = not related
		- -1 = moves oppositie, perfectly related
-- WHY:
	- highly correlated features carry redundant information
		- confuses models, inflate importance of redundant features
		- increase dimensionality unnecessarily, slows training and may cause overfitting
	- cleaner feature set, less redundancy, more stable
-- HOW:
	- if correlation > 0.9, consider dropping

3. better feature engineering/selection
- modify feature engineering to calculate more potentially useful feature based on data and hypotheses, avoid technical indicators, focus on onchain data, think about relationships in my data

4. strategy logic
FINDINGS (MY PROCESS):
- signal mapping is part of turning hypotheses to data
- everytime i got a decent backtest result, i would tune the signal mapping first, get a better backtest result, only to find out that the trade frequency sucks balls in the forwardtest, which wastes a lot of time
- i was beginning to suspect that the backtest HMM states were not relevant in the forwardtests, so my initial plan to solve that issue, is to isolate and map all states to buy/sell signals, and immediately run both backtest and forward test to see the trade frequency whether the trades were registered according to its states
- then i stumble across state distribution comparison, essentially doing normalization without realizing:
-- WHAT:
	- the percentage of the time a model assigned each state
	- considered as distribution normalization
-- WHY:
	- makes model more robust, ensures the states show up in both backtest and forwardtest
	- able to detect overfitting easily, understood that market regimes are different
-- HOW:
	- eg: with 3 states
	- states = [0, 2, 1, 1, 0, 2, 2, 0]
	- pd.Series(states).value_counts(normalize=True).sort_index()
	- result:
		- 0: 37.5%
		- 1: 25%
		- 2: 37.5%
- this largely helped when I identified a great backtest result and forwardtest result, only to realize that my forwardtest result to skyrocket its trade frequency, meaning the model overfit to specific states, it didnt make sense because forwardtest period was only 1 year while backtest was 3 years
- then i also realized the backtest hmm_states that i trained were highly irrelevant to the forwardtest hmm_states because of my cookie-cutter shitty features that i was using this entire time, and i randomly tested different features and still couldnt get the state_difference_threshold to be lower than 20%, it was always 26-60+ i was starting to acknowledge the flaw in my features, i had to do something about it
- realizing that knowing what dataset to use requires domain knowledge and recognizing that i have extremely limited time for this hackathon, i decided to not change my dataset, and instead use gemini-deep-research to help me find potential features using the same dataset, used the ATC workshop pdf as reference to prompt, then let it do its thing
- i ended up with nearly 40 features, and i just ran the model with those features, since i had basic feature selection using correlation matrix, the output identified 8 most significant features used, i uncommented the rest of them and only focused on the 8, and it gave me a 13.72% state_difference, finally passing the threshold
- and honestly, i accidentally found the best performance so far with my old signal mapping, of course i did analyze the states and compare what regime the states represent, i did try to fine tune it, but it all ended up worse, so i stuck with that accidental configuration that gave me that stellar performance

potential improvements:
1. presentation
- visualize data, equity curve
- save the config of the best result?
- architecture diagram
- cumulative PnL graph
- Sortino Ratio and time to Recover from MDD
- Be clear on the HOWs, specifically in the context of my own model

finals requirements
- working framework
- documentation
- a strategy developed using the framework
