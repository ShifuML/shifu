/*
 * Copyright [2012-2014] PayPal Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.container;

import java.util.ArrayList;
import java.util.List;

/**
 * Model initialization input
 */
public class ModelInitInputObject {

    private List<Double> weights;
    private int numWeights;
    private List<Integer> sampleIndex;
    private int numSample;
    private List<Integer> trainSetIndex;
    private int numTrainSet;
    private List<Integer> validSetIndex;
    private int numValidSet;

    public ModelInitInputObject() {
        numWeights = 0;
        numSample = 0;
        numTrainSet = 0;
        numValidSet = 0;

        weights = new ArrayList<Double>();
        sampleIndex = new ArrayList<Integer>();
        trainSetIndex = new ArrayList<Integer>();
        validSetIndex = new ArrayList<Integer>();
    }

    /**
     * @return the trainSetIndex
     */
    public List<Integer> getTrainSetIndex() {
        return trainSetIndex;
    }

    /**
     * @param trainSetIndex the trainSetIndex to set
     */
    public void setTrainSetIndex(List<Integer> trainSetIndex) {
        this.trainSetIndex = trainSetIndex;
    }

    /**
     * @return the numTrainSet
     */
    public int getNumTrainSet() {
        return numTrainSet;
    }

    /**
     * @param numTrainSet the numTrainSet to set
     */
    public void setNumTrainSet(int numTrainSet) {
        this.numTrainSet = numTrainSet;
    }

    /**
     * @return the validSetIndex
     */
    public List<Integer> getValidSetIndex() {
        return validSetIndex;
    }

    /**
     * @param validSetIndex the validSetIndex to set
     */
    public void setValidSetIndex(List<Integer> validSetIndex) {
        this.validSetIndex = validSetIndex;
    }

    /**
     * @return the numValidSet
     */
    public int getNumValidSet() {
        return numValidSet;
    }

    /**
     * @param numValidSet the numValidSet to set
     */
    public void setNumValidSet(int numValidSet) {
        this.numValidSet = numValidSet;
    }

    /**
     * @return the weights
     */
    public List<Double> getWeights() {
        return weights;
    }

    /**
     * @param weights the weights to set
     */
    public void setWeights(List<Double> weights) {
        this.weights = weights;
    }

    /**
     * @return the numWeights
     */
    public int getNumWeights() {
        return numWeights;
    }

    /**
     * @param numWeights the numWeights to set
     */
    public void setNumWeights(int numWeights) {
        this.numWeights = numWeights;
    }

    /**
     * @return the sampleIndex
     */
    public List<Integer> getSampleIndex() {
        return sampleIndex;
    }

    /**
     * @param sampleIndex the sampleIndex to set
     */
    public void setSampleIndex(List<Integer> sampleIndex) {
        this.sampleIndex = sampleIndex;
    }

    /**
     * @return the numSample
     */
    public int getNumSample() {
        return numSample;
    }

    /**
     * @param numSample the numSample to set
     */
    public void setNumSample(int numSample) {
        this.numSample = numSample;
    }
}
