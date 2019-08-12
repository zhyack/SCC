# SCC (Skilled Chess Commentator)

Codes of models and data processing for paper "Automated Chess Commentator Powered by Neural Chess Engine, **Hongyu Zang**, **Zhiwei Yu** and Xiaojun Wan, **ACL2019**" ([pdf](https://www.aclweb.org/anthology/P19-1597)) will be released in this repo.  
([Zhiwei](https://github.com/ArleneYuZhiwei) and me contribute equally in this paper.)

## Updating
* Data preparation
    * Make sure you follow [this repo](https://github.com/harsh19/ChessCommentaryGeneration) to get the basic dataset in `data/crawler` 
    * Ask the permission from Jhamtani et.al for the distribution of `annotation2.tsv`, `rules.txt`, and their processed dataset with `[train/test/valid]_[0/1/2].en` for data pre-processing.
    * Put the above file into `data/mycrawler/data/`, and run the scripts (to be detailed) in `data/mycrawler/` to get `data/mycrawler/data/[train/valid/test].pickle`
* Experiment environments
    * Python 3.5, Tensorflow 1.8
    * Necessary: [GAC](https://github.com/harsh19/ChessCommentaryGeneration), [alpha-zero-general](https://github.com/suragnair/alpha-zero-general)
    * In case you need: [Arena](http://www.playwitharena.de/), [deep-pink](https://github.com/erikbern/deep-pink), [sunfish](https://github.com/thomasahle/sunfish)
* Training
    * to be detailed
* Test the chess engine
    * Install Arena in `arena/`. Get `sunfish` and `deep-pink` in corresponding folders. Replace with files already in the folders.
    * Download the checkpoint use the links in `chess-agent/SCC/links`
    * Run Arena and compete with our model by adding `chess-agent/engine` into Arena engines. (to be detailed)
* Reproduce the results
    * Get `Data preparation` Done.
    * Download the checkpoint use the links in `chess-agent/SCC/links`
    * `cd chess-agent/`
    * `python main.py -c mixall`

## Note

* The codes and README is still updating, more details will be cleared. Please be patient.

* You can check previous most related work about Chess Commentary ([GAC](https://github.com/harsh19/ChessCommentaryGeneration)).

* We also use the dataset provided by GAC. You may need to require the permission of distributions of the processed data and scripts from Jhamtani et al. (see previous link).

* We build our chess agent on [alpha-zero-general](https://github.com/suragnair/alpha-zero-general). If you are interested, you can learn and extend this project.
