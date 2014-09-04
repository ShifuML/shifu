/**
 * Copyright [2012-2014] eBay Software Foundation
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
package ml.shifu.plugin.spark.stats.unitstates;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import ml.shifu.core.container.NumericalValueObject;
import ml.shifu.plugin.spark.stats.SerializedNumericalValueObject;


/**
 * Maintains a reservoir sample.
 */

public class RSampleUnitState<T> implements Serializable {

    
    private static final long serialVersionUID = 1L;
    protected List<T> samples;
    protected int maxSize;
    protected Random intRand;
    protected int n;
    
    public RSampleUnitState(int maxSize) {
        this.samples= new ArrayList<T>();
        this.maxSize= maxSize;
        this.intRand= new Random();
        this.n= 0;
    }
    
    
    public void merge(RSampleUnitState<T> otherState) throws Exception {
        /*
         * find number of elements to be taken from this and otherState, then use systematic sampling
         * to sample that number of elements, and combine both samples into this.samples 
         */
        
        int nSamplesForThis= (int)(maxSize * ((double)(this.n)/(this.n + otherState.n)));
        int nSamplesForOther= maxSize - nSamplesForThis;
        this.samples= systematicSample(nSamplesForThis);
        this.samples.addAll(otherState.systematicSample(nSamplesForOther));
        this.n+= otherState.n;        
        n+= otherState.n;
    }
    
    public List<T> systematicSample(int numSamples) {
        // samples n elements from the reservoir sample using systematic sampling
        // select nSamplesForThis from this.samples
        ArrayList<T> newSamples= new ArrayList<T>();
        double interval= (double)this.samples.size()/numSamples;
        Random rand= new Random();
        int size= this.samples.size();
        int nextIntDist, currIndex= 0;
        for(int i=0; i < numSamples; i++) {
            nextIntDist= (int)(rand.nextDouble()*interval) + 1;
            currIndex+= nextIntDist;
            currIndex%= size;
            newSamples.add(samples.get(currIndex));
        }
        return newSamples;
    }
    
    
    public List<T> getSamples() {
        return samples;
    }
        
    
    public void addSample(T sample) {
        n++;
        if(samples.size() < maxSize)
            samples.add(sample);
        else {
            // inclusive range for nextInt()
            int position= intRand.nextInt(n+1);
            if(position < maxSize ) {
                // include element in sample
                samples.set(position, sample);
            }
        }
    }
    
    
}
