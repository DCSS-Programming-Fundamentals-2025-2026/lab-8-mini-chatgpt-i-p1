using System.Text.Json;
public class NGramModel
{
    private float[][] _probs;
    private NGramCounts _counts;

    public NGramModel(int vocabSize){
        _probs = new float[vocabSize][];

        for(int i = 0; i < vocabSize; i++)
        {
            _probs[i] = new float[vocabSize];
        }

        _counts = new NGramCounts(vocabSize);
    }

    public void Train(ReadOnlySpan<int> tokens)
    {
        _counts = new NGramCounts(_probs.Length);
        _counts.CountBigrams(tokens);

        for(int i = 0; i < _probs.Length; i++)
        {
            float rowSum = _counts.BigramCounts[i].Sum();

            if(rowSum > 0)
            {
                for(int j = 0; j < _probs[i].Length; j++)
                {
                    _probs[i][j] = _counts.BigramCounts[i][j] / rowSum;
                }
            }
        }
    }

    public float[] NextTokenScores(ReadOnlySpan<int> context)
    {
        if(context.IsEmpty)
        {
            float[] uniform = new float[_probs.Length];
            for(int k = 0; k < uniform.Length; k++) {
                uniform[k] = 1f / _probs.Length;
            }
            return uniform;
        }

        int last = context[context.Length-1];

        float[] copy = new float[_probs[last].Length];
        Array.Copy(_probs[last], copy, copy.Length);

        return copy;
    }

    public NGramPayloadMapper GetPayloadForCheckpoint()
    {
        NGramPayloadMapper _container = new NGramPayloadMapper()
        {
            BigramProbs = _probs
        };

        return _container;
    }

    public void FromPayload(JsonElement payload)
    {
        JsonElement probsArray = payload.GetProperty("BigramProbs");

        int vocabSize = probsArray.GetArrayLength();

        _probs = new float[vocabSize][];

        int i = 0;

        foreach(JsonElement rowElement in probsArray.EnumerateArray())
        {
            int rowLength = rowElement.GetArrayLength();
            _probs[i] = new float[rowLength];

            int j = 0;

            foreach(JsonElement colElement in rowElement.EnumerateArray())
            {
                _probs[i][j] = colElement.GetSingle();
                j++;
            }
            i++;
        }
    }
}