using Lib.Corpus.Configuration;
using Lib.Corpus.Processing;

namespace Lib.Corpus.Domain
{
    public class CorpusLoader : ICorpusLoader
    {
        private readonly CorpusTextNormalizer textNormalizer;
        private readonly CorpusSplitter corpusSplitter;
        private readonly IFileSystem defaultFileSystem;
        private readonly string Version = "1.0.0";
    
        public string GetContractFingerprint()
        {
            return Version;
        }

        public CorpusLoader(CorpusTextNormalizer textNormalizer, CorpusSplitter corpusSplitter, IFileSystem defaultFileSystem)
        {
            this.textNormalizer = textNormalizer;
            this.corpusSplitter = corpusSplitter;
            this.defaultFileSystem = defaultFileSystem;
        }

        public CorpusClass Load(string path, CorpusLoadOptions? options)
        {
            if (options == null)
            {
                options = new CorpusLoadOptions();
            }

            bool exist = defaultFileSystem.Exists(path);
            string content;

            if (exist)
            {
                content = defaultFileSystem.ReadAllText(path);
            }
            else
            {
                content = options.FallBack;
            }

            return LoadFromText(content, options);
        }

        public CorpusClass LoadFromText(string text, CorpusLoadOptions? options)
        {
            if (options == null)
            {
                options = new CorpusLoadOptions();
            }

            text = textNormalizer.Normalize(options.LowerCase, text);

            string[] parts = corpusSplitter.Splitter(text, options.ValidateFraction);
            string TrainText = parts[0];
            string ValidatePart = parts[1];

            return new CorpusClass(TrainText, ValidatePart);
        }
    }
}