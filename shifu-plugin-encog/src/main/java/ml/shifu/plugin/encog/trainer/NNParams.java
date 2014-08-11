package ml.shifu.plugin.encog.trainer;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)

public class NNParams {


    private Double splitRatio;
    private List<HiddenLayer> hiddenLayers;
    private String algorithm;
    private Double learningRate = null;
    private Integer numEpochs = 100;


    public Integer getNumEpochs() {
        return numEpochs;
    }

    public void setNumEpochs(Integer numEpochs) {
        this.numEpochs = numEpochs;
    }


    public Double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(Double learningRate) {
        this.learningRate = learningRate;
    }


    public String getAlgorithm() {
        return algorithm;
    }

    public void setAlgorithm(String algorithm) {
        this.algorithm = algorithm;
    }


    public Double getSplitRatio() {
        return splitRatio;
    }

    public void setSplitRatio(Double splitRatio) {
        this.splitRatio = splitRatio;
    }

    public List<HiddenLayer> getHiddenLayers() {
        return hiddenLayers;
    }

    public void setHiddenLayers(List<HiddenLayer> hiddenLayers) {
        this.hiddenLayers = hiddenLayers;
    }


}