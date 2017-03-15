/*
 * Copyright [2013-2015] PayPal Software Foundation
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
package ml.shifu.shifu.guagua;

import org.apache.hadoop.mapreduce.lib.input.FileSplit;

import ml.shifu.guagua.hadoop.io.GuaguaInputSplit;

public class ShifuInputSplit extends GuaguaInputSplit {
    private boolean isCrossValidation;
    
    
    public ShifuInputSplit(boolean isMaster,boolean isCrossValidation, FileSplit fileSplit) {
        super(isMaster, fileSplit);
        this.isCrossValidation = isCrossValidation;
    }
    
    public ShifuInputSplit(boolean isMaster, boolean isCrossValidation,FileSplit... fileSplits) {
        super(isMaster,fileSplits);
        this.isCrossValidation = isCrossValidation;
    }

    public boolean isCrossValidation() {
        return isCrossValidation;
    }
    public void setCrossValidation(boolean isCrossValidation) {
        this.isCrossValidation = isCrossValidation;
    }

}
