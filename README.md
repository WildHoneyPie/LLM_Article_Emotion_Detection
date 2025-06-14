# Music Emotion Analysis (Arousal-Valence)

This project analyzes music files to detect emotional content using the Circumplex Model of Affect, which measures:
- **Valence**: The positivity/negativity of the emotion (-1 to 1)
- **Arousal**: The energy level of the emotion (-1 to 1)

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd music-arousal-valence
```

2. Install required packages:
```bash
pip install essentia-tensorflow numpy seaborn pandas matplotlib
```

3. Download the required model files:
```bash
mkdir models
cd models
curl -L -o msd-musicnn-1.pb "https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.pb"
curl -L -o audioset-vggish-3.pb "https://essentia.upf.edu/models/feature-extractors/vggish/audioset-vggish-3.pb"
curl -L -o deam-audioset-vggish-1.pb "https://essentia.upf.edu/models/classification-heads/deam/deam-audioset-vggish-1.pb"
curl -L -o deam-msd-musicnn-1.pb "https://essentia.upf.edu/models/classification-heads/deam/deam-msd-musicnn-1.pb"
curl -L -o emomusic-audioset-vggish-1.pb "https://essentia.upf.edu/models/classification-heads/emomusic/emomusic-audioset-vggish-1.pb"
curl -L -o emomusic-msd-musicnn-1.pb "https://essentia.upf.edu/models/classification-heads/emomusic/emomusic-msd-musicnn-1.pb"
curl -L -o muse-audioset-vggish-1.pb "https://essentia.upf.edu/models/classification-heads/muse/muse-audioset-vggish-1.pb"
curl -L -o muse-msd-musicnn-1.pb "https://essentia.upf.edu/models/classification-heads/muse/muse-msd-musicnn-1.pb"
cd ..
```

## Usage

1. Place your audio file (e.g., Pads.wav) in the project directory

2. Run the analysis script:
```bash
python analyze.py
```

The script will:
- Analyze the audio using multiple model combinations
- Generate visualizations for each analysis
- Save results to `analysis_results.json`

## Output

The analysis provides:
- Valence scores (-1 to 1, where 1 is positive)
- Arousal scores (-1 to 1, where 1 is high energy)
- Visual plots showing the emotional position in the arousal-valence space
- JSON file with detailed results

## Models Used

The analysis uses multiple model combinations:
- **MusiCNN-based**:
  - msd-musicnn + emomusic
  - msd-musicnn + deam
  - msd-musicnn + muse
- **VGGish-based**:
  - audioset-vggish + emomusic
  - audioset-vggish + deam
  - audioset-vggish + muse

## License

These models are part of [Essentia Models](https://essentia.upf.edu/models.html) made by [MTG-UPF](https://www.upf.edu/web/mtg/) and are publicly available under [CC by-nc-sa](https://creativecommons.org/licenses/by-nc-sa/4.0/) and commercial license.
