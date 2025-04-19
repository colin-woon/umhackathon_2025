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
- Using State Distribution check to break down backtest period into chunks now, it identifies how consistent the hidden states appear over different periods of time, helps diagnose the model robustness, but right now its usage its questionable
- Using T-test to statiscally prove our features are solid
-- WHAT:
	- a way to check if two groups of numbers are really different from each other, or if the difference is just random luck.
-- WHY:
	- To quantify whether the difference in feature distributions between two classes is statistically significant — supporting that our features are discriminative and meaningful.

	- A high T-statistic (4.44) and an extremely low p-value (~0.0000) in our backtest, and T = 2.94, p = 0.0034 in our forward test, suggest that our strategy's returns are not due to random chance. This statistically confirms that our features are capturing true predictive signals, not noise.
-- HOW:
	- To show our features have meaningful predictive power:
		- P-value < 0.05, reject null hypotheses, means its not by chance, backed by statistics
		- P-value >= 0.05, fail to reject null hypotheses, could be by chance, backed by statistics
		- T-statistic --  Some days you win, some days you lose. The T-statistic considers how much these daily results jump around (their standard deviation/standard error), Generally, the larger the absolute value of the T-statistic (further from zero), the smaller the P-value will be.

- Using Permutation of shuffled signals to prove our features dont work for bad conditions, null hypotheses
-- WHAT:
	- randomly shuffles the 'signal' values in our trading strategy data. This simulates a world where the features have no predictive power, i.e., a null hypothesis where patterns in the data are purely by chance.
-- WHY:
	- We do this to validate that our features and signals are meaningful and not just overfitting noise.
	- if strategy performace:
		- fails badly, means original signals were important and structured
		- stays good, strategy might be exploting noise or data leakage, not real patterns
-- HOW:
	- Copy the original strategy signals DataFrame.
	- Randomly shuffle the 'signal' column (this breaks any real-time structure or pattern).
	- Run the backtest on the shuffled data to see how it performs under this null condition.
	- Repeat the shuffle and backtest process (e.g., 1000 times) to get a distribution of Sharpe ratios from random noise.
	- Comparing real strategy's Sharpe against this distribution:
		- If your Sharpe > 95th percentile of the shuffled distribution → statistically significant, not by chance.
		- If your Sharpe < 95th percentile, falls within the random range → possibly just random noise.

3. better feature engineering/selection
- modify feature engineering to calculate more potentially useful feature based on data and hypotheses, avoid technical indicators, focus on onchain data, think about relationships in my data
- UPDATE: after everything, we still havent really addressed this issue, all we have is the validation of our features, but instead of theory, we are using statistics to back up our data
- all the features were AI generated through gemini deep-research with specific prompts.

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
- UPDATE: i found out by doing state distribution this way, i made a grave mistake, i am essentially taking a peek into the forwardtest data, and fine tuning my signals again, which i was not supposed to do. essentially its like i cheated and got tips to the answers of the test, and fine-tuned my signals again.
- so the state distribution was refactored to split the backtest period into chunks, and do comparisons based on those chunks, i found that it got 35%, which is a lot more than the 13% that i got initially when i compare the backtest and forward test, arguably does this mean that my model has overfit? or does it mean that i dont need such a low distribution difference to get a good performance result
- we will have to actually develop a strategy with hourly timeframe and test again now, since disclamer is that our results are basically not that valid. its like we were given tips already
- we had an idea of testing every possible permutations of signal maps to get the best results, but the code is likely to be computational expensive, thats why we implemented:
	- Bayesian Optimization
	-- WHAT:
		- a smart search method, it tries some signal maps, see how well they perform
	-- WHY:
		- builds a statistical model of which maps are likely to be good, then uses it to decide which maps to try next
		- aims to find the best map much faster
	-- HOW:
		- Define the Search Space: You tell the optimizer what it can change.
		- Define the Goal (Objective Function)
			- takes a proposed signal map
			- runs your backtest simulation
			- calculates a score (like Sharpe Ratio or Total Return)
		- Run the Optimizer
			- It first tries a few random signal maps to get a rough idea.
			- enters a loop (can also stop early if performance doesn't improve for a set number of patience iterations.) :
				- uses the scores  to build its internal statistical model (the Gaussian Process) guessing how good any map might be.
				- uses an "acquisition function", to decide which untested map looks most promising to try next (balance between exploiting already good results and exploration)
				- calls your objective_function with this promising map to get its actual score.
				- updates its internal model with this new result.
		- Get the Best Result
- using this signal generator, we still tested with the model that we previously did n_state distribution with forward test data before, and it went from a whopping 1000%+ return improved to 3000%+, and SR went from 2.7 to 2.9
- currently we are still stuck with trying to find a robust strategy for hourly timeframe, and out daily timeframe strategy is now kinda invalid

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
