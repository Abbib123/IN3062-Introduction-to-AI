# README

-- Current file is a test file to add to the repo, will be updated as the cw progresses
-- Open in jupyter/spyder/VSCode
-- Run
-- MAY NEED TO BE RUN VIA HYPERION IN THE LATTER STAGES DUE TO COMPUTATIONAL REQUIREMENTS
-- Plan for future work:
	-- Clean dataset:
		-- Remove ID (unecessary)
		-- Remove columns for each possible disease (only need the label)
		-- Remove Filepath
		-- remove keywords (label needed only)
		-- Remove Target (Only label)
		-- Should be left with Age/Sex/Label/Filename
		-- Clean Filename to leave only the letter via removing the [,',',]
		-- Transform data if we don't have enough to fluff up data
	-- Algos for Accuracy:
		-- CNN
		-- K-Means Clustering
		-- SVM
		-- Activation Maps
		-- Confusion Matrix for test results
	-- Beginning:
	Binary classification:
		-- 1-1 Normal vs Diabetes (example)