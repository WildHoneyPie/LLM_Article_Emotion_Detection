import json
import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from essentia.standard import (
    MonoLoader,
    TensorflowPredictVGGish,
    TensorflowPredictMusiCNN,
    TensorflowPredict2D,
)

# Setup paths
MODELS_HOME = "models"  # Make sure this directory exists and contains the model files

def setup_models():
    """Load the model into memory and create the Essentia network for predictions"""
    musicnn_graph = os.path.join(MODELS_HOME, "msd-musicnn-1.pb")
    vggish_graph = os.path.join(MODELS_HOME, "audioset-vggish-3.pb")

    sample_rate = 16000

    loader = MonoLoader()
    embeddings = {
        "msd-musicnn": TensorflowPredictMusiCNN(
            graphFilename=musicnn_graph,
            output="model/dense/BiasAdd",
            patchHopSize=187,
        ),
        "audioset-vggish": TensorflowPredictVGGish(
            graphFilename=vggish_graph,
            output="model/vggish/embeddings",
            patchHopSize=96,
        ),
    }

    input_layer = "flatten_in_input"
    output_layer = "dense_out"
    classifiers = {}

    datasets = ("emomusic", "deam", "muse")
    for dataset in datasets:
        for embedding in embeddings.keys():
            classifier_name = f"{dataset}-{embedding}"
            graph_filename = os.path.join(MODELS_HOME, f"{classifier_name}-1.pb")
            classifiers[classifier_name] = TensorflowPredict2D(
                graphFilename=graph_filename,
                input=input_layer,
                output=output_layer,
            )

    return loader, embeddings, classifiers, sample_rate

def analyze_audio(audio_path, embedding_type="msd-musicnn", dataset="emomusic"):
    """Analyze an audio file and return arousal and valence values"""
    loader, embeddings, classifiers, sample_rate = setup_models()

    print("Loading audio...")
    loader.configure(
        sampleRate=sample_rate,
        filename=audio_path,
        resampleQuality=4,
    )
    waveform = loader()

    embeddings_result = embeddings[embedding_type](waveform)

    classifier_name = f"{dataset}-{embedding_type}"
    results = classifiers[classifier_name](embeddings_result)
    results = np.mean(results.squeeze(), axis=0)

    # Manual normalization (1, 9) -> (-1, 1)
    results = (results - 5) / 4

    valence = results[0]
    arousal = results[1]

    return valence, arousal

def plot_results(valence, arousal, title):
    """Create a visualization of the arousal-valence results"""
    sns.set_style("darkgrid")
    g = sns.lmplot(
        data=pd.DataFrame({"arousal": [arousal], "valence": [valence]}),
        x="valence",
        y="arousal",
        markers="x",
        scatter_kws={"s": 100},
    )

    g.set(ylim=(-1, 1))
    g.set(xlim=(-1, 1))
    plt.plot([-1, 1], [0, 0], linewidth=1.5, color="grey")
    plt.plot([0, 0], [-1, 1], linewidth=1.5, color="grey")
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.15)
    plt.title(title)

    # Save the plot
    plt.savefig("arousal_valence_plot.png")
    print("Plot saved as 'arousal_valence_plot.png'")

def main():
    audio_file = "Pads.wav"
    
    # Analyze with different model combinations
    model_combinations = [
        ("msd-musicnn", "emomusic"),
        ("msd-musicnn", "deam"),
        ("msd-musicnn", "muse"),
        ("audioset-vggish", "emomusic"),
        ("audioset-vggish", "deam"),
        ("audioset-vggish", "muse"),
    ]

    results = {}
    
    for embedding_type, dataset in model_combinations:
        print(f"\nAnalyzing with {embedding_type} embedding and {dataset} dataset...")
        valence, arousal = analyze_audio(audio_file, embedding_type, dataset)
        
        results[f"{embedding_type}_{dataset}"] = {
            "valence": float(valence),
            "arousal": float(arousal)
        }
        
        print(f"Results for {embedding_type} + {dataset}:")
        print(f"Valence: {valence:.3f}")
        print(f"Arousal: {arousal:.3f}")
        
        # Create visualization for each combination
        plot_results(valence, arousal, f"{embedding_type} + {dataset}")

    # Save all results to JSON
    with open("analysis_results_pads.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nAll results saved to 'analysis_results_pads.json'")

if __name__ == "__main__":
    main() 