/*
 * Copyright [2013-2015] eBay Software Foundation
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
package ml.shifu.shifu.core.converge;

import java.util.Map;
import java.util.Map.Entry;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author xiaobzheng
 * interface for training convergence judge.
 */
public abstract class AbstractConvergeJudger {
    
    /**
     * enum type that indicating where the input error come from.
     * TRAIN means error coming from training data, while TEST coming from validation data
     */
    public static enum ERROR_OWNER {
        TRAIN, TEST, PRE_TRAIN, PRE_TEST;
    }
    
    protected Double trainError = Double.valueOf(0.0);
    
    protected Double testError = Double.valueOf(0.0);
    
    protected Double preTrainError = Double.valueOf(0.0);
    
    protected Double preTestError = Double.valueOf(0.0);
    
    protected Double threshold = Double.valueOf(10e-4);
    
    /**
     * Log for abstract ConvergeJudger
     */
    protected static Logger LOG = LoggerFactory.getLogger(AbstractConvergeJudger.class);

    /**
     * @param errors Map that contains errors result generating from 
     *        specific data set type (types defined in {@link ERROR_OWNER})
     * @return if it's converged then return true, else return false.
     * @throws Exception
     */
    public boolean isConverge(Map<ERROR_OWNER, Double> errors, Double threshold) throws Exception {
        LOG.info("Initial judger inputs.");
        init(errors, threshold);
        
        LOG.info("Start judge Convergence:");
        boolean judgeResult = false;
        
        LOG.info("Check input errors first.");
        checkInput();

        LOG.info("Compute Convergence judging result");
        judgeResult = judge();

        LOG.info("Convergence judge finish , result is : " + judgeResult);
        return judgeResult;
    }
    
    //TODO xiaobin: add other overloading isConverged function. 
    
    /**
     * @param errors contains errors result generating from 
     *        specific data set type (types defined in {@link ERROR_OWNER})
     * @param threshold value for convergence judge.
     */
    protected void init(Map<ERROR_OWNER, Double> errors, Double threshold) {
       if (errors != null) {
           for (Entry<ERROR_OWNER, Double> error : errors.entrySet()) {
               setErrorByOwner(error.getKey(), error.getValue());
           }
           
           setThreshold(threshold);
       } else {
           throw new IllegalArgumentException("Couldn't initial ConvergeJudger with null point Map instance."); 
       }
    }
    
    /**
     * @return if it's converged then return true, else return false.
     */
    protected abstract boolean judge();
    
    /**
     * <p/>
     * Check input error results. if there is invalid error result, then it will
     * throw {@link Exception}
     * <p/>
     */
    protected void checkInput() {
        //TODO xiaobin : do some check for input errors including training and validation error.
        //     not implemented now.
        boolean checkResult = true;
        if (!checkResult) {
            throw new IllegalArgumentException("Input incorrect");
        } else {
            LOG.info("Input correct !");
        }
    }

    private void setErrorByOwner(ERROR_OWNER owner, Double value) {
        switch(owner) {
            case TRAIN:
                setTrainError(value);
                break;
            case TEST:
                setTestError(value);
                break;
            case PRE_TRAIN:
                setPreTrainError(value);
                break;
            case PRE_TEST:
                setPreTestError(value);
                break;
            default:
                break;
        }
    }
    
    public Double getTrainError() {
        return trainError;
    }

    public void setTrainError(Double trainError) {
        this.trainError = trainError;
    }

    public Double getTestError() {
        return testError;
    }

    public void setTestError(Double testError) {
        this.testError = testError;
    }

    public Double getPreTrainError() {
        return preTrainError;
    }

    public void setPreTrainError(Double preTrainError) {
        this.preTrainError = preTrainError;
    }

    public Double getPreTestError() {
        return preTestError;
    }

    public void setPreTestError(Double preTestError) {
        this.preTestError = preTestError;
    }

    public Double getThreshold() {
        return threshold;
    }

    public void setThreshold(Double threshold) {
        this.threshold = threshold;
    }
}
