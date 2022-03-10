# Chinese News Summarization

## Task Description
- Input: News content
```
從小就很會念書的李悅寧， 在眾人殷殷期盼下，以榜首之姿進入臺大醫學院， 但始終忘不了對天文的熱情。大學四年級一場遠行後，她決心遠赴法國攻讀天文博士。 從小沒想過當老師的她，再度跌破眾人眼鏡返台任教，......
```

- Output: News Title
```
榜首進台大醫科卻休學，27歲拿到法國天文博士，李悅寧跌破眾人眼鏡返台任教
```

## Data Format
- Train: 21710 articles from 2015-03-02 to 2021-01-13
- Public: 5494 articles from 2021-01-14 to 2021-04-10
- [dataset link](https://drive.google.com/file/d/186ejZVADY16RBfVjzcMcz9bal9L3inXC/view)

### Example
```
{
    "date_publish": "2015-03-02 00:00:00",
    "title": "榜首進台大醫科卻休學，27歲拿到法國天文博士 李悅寧跌破眾人眼鏡返台任教",
    "source_domain": "udn.com",
    "maintext": "從小就很會念書的李悅寧，在眾人殷殷期盼下，以榜首之姿進入台大醫學院，但始終忘不了對天文的熱情，..."
}
```

## Instructions

To download the model, run
```
bash download.sh
```

To generate titles for test data, run
```
bash run.sh /path/to/input.jsonl /path/to/output.jsonl
```

For evalutation, refer to this [repo](https://github.com/ntu-adl-ta/ADL21-HW3).

To train the model, run
```
python train.py /path/to/trainData /path/to/saveModel
```