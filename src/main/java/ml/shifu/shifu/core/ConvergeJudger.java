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
package ml.shifu.shifu.core;

/**
 * Class for training convergence judging. 
 * 
 * <p> Initialization value for train, test error value is <code>{@link Double#POSITIVE_INFINITY}</code> and
 * 0.0d for threshold value. So you should set train error, test error and threshold by their setter function 
 * before any judging operations. After that you can use <code>{@link ConvergeJudger#isConverged()}</code>
 * to get convergence result. For example,
 * 
 * <pre>
 * ConvergeJudger judger = new ConvergeJudger();
 * judger.setTrainErr(1.0);
 * judger.setTestErr(2.0);
 * judger.setThreshold(0.01);
 * boolean result = judger.isConverged();
 * </pre>
 * 
 * <p> <code>{@link ConvergeJudger#isConverged()}</code> returns true if (train_err + test_err) / 2 <= threshold,
 * else false.
 * 
 * <p> If you just want get the average error value, use <code>{@link ConvergeJudger#CalculateAvgErr()}</code> 
 * after judging setting complete. For example,
 * 
 * <pre>
 * judger.setTrainErr(1.0);
 * judger.setTestErr(2.0);
 * Double avgErr = judger.CalculateAvgErr();
 * </pre>
 * 
 * <p> Both <code>{@link ConvergeJudger#isConverged()}, {@link ConvergeJudger#CalculateAvgErr()}</code> will
 * update this.currentAvgErr. You can directly get latest updated average error value by using 
 * {@link ConvergeJudger#getCurrentAvgErr()}</code> without calculation.
 * 
 * <p> Use <code>{@link ConvergeJudger#reset()}</code> to restore both errors, threshold and current average error
 *  to their initialization value.
 * 
 * <p> You can also get currently input error and threshold setting by using
 * <code>{@link ConvergeJudger#getJudgeSetting()}</code>.
 * 
 * @author xiaobzheng (zheng.xiaobin.roubao@gmail.com)
 */

public class ConvergeJudger {

    private Double trainErr = Double.POSITIVE_INFINITY;
    
    private Double testErr = Double.POSITIVE_INFINITY;
    
    private Double threshold = Double.valueOf(0.0);

    private Double currentAvgErr = Double.POSITIVE_INFINITY;
    
    /**
     * Compute average error using preset training error and testing error. If the average value 
     * is less than or equal to the preset threshold, it will return true, else false.
     * 
     * <p> This operation will upudate current average error.
     * 
     * @return convergence judging result.
     */
    public boolean isConverged() {
        CalculateAvgErr();

        return currentAvgErr.compareTo(threshold) <= 0 ? true : false;
    }

    /**
     * Generate currently judge setting with train error, test error and threshold.
     * Setting String's format as,
     * 
     * <pre>
     * "[trainErr: 15.026540, testErr: 16.023540, threshold: 0.00001]"
     * </pre>
     * 
     * @return String that presenting currently input setting.
     */
    public String getJudgeSetting() {
        return String.format("[trainErr: %f, testErr: %f, threshold: %f]", trainErr, testErr, threshold);
    }
    
    /**
     * Calculate average value of preset train error and test error. 
     * 
     * <p> This operation will upudate current average error.
     * 
     * @return (trainErr + testErr) / 2
     */
    public Double CalculateAvgErr() {
        setCurrentAvgErr((trainErr + testErr) / 2);
        return currentAvgErr;
    }
    
    /**
     * Reset judger input setting. Set <code>{@link Double#POSITIVE_INFINITY}</code> to train error, test error and 
     * current average error. Set 0.0d to threshold.
     */
    public void reset() {
        setTrainErr(null);
        setTestErr(null);
        setThreshold(null);
        setCurrentAvgErr(null);
    }
    
    /**
     * Get average error updated by latest operation: <code>{@link ConvergeJudger#isConverged()}</code>
     * or <code>{@link ConvergeJudger#CalculateAvgErr()}</code>.
     * 
     * <p> Its Initial value is <code>{@link Double#POSITIVE_INFINITY}</code>.
     * 
     * @return current average error.
     */
    public Double getCurrentAvgErr() {
        return currentAvgErr;
    }

    /**
     * If input error is null, this.currentAvgErr will be assigned <code>{@link Double#POSITIVE_INFINITY}</code>.
     * 
     * @param currentAvgErr input average error.
     */
    private void setCurrentAvgErr(Double currentAvgErr) {
        this.currentAvgErr = currentAvgErr == null ? Double.POSITIVE_INFINITY : currentAvgErr;
    }

    public Double getTrainErr() {
        return trainErr;
    }

    /**
     * If input train err is null, this.trainErr will be assigned <code>{@link Double#POSITIVE_INFINITY}</code>.
     * 
     * @param trainErr input train error.
     */
    public void setTrainErr(Double trainErr) {
        this.trainErr = trainErr == null ? Double.POSITIVE_INFINITY : trainErr;
    }

    public Double getTestErr() {
        return testErr;
    }

    /**
     * If input test err is null, then this.testErr will be assigned <code>{@link Double#POSITIVE_INFINITY}</code>.
     * 
     * @param testErr input test error.
     */
    public void setTestErr(Double testErr) {
        this.testErr = testErr == null ? Double.POSITIVE_INFINITY : testErr;
    }

    public Double getThreshold() {
        return threshold;
    }

    /**
     * If input threshold is null, then this.threshold will be assigned 0.0d.
     * 
     * @param threshold input threshold.
     */
    public void setThreshold(Double threshold) {
        this.threshold = threshold == null ? Double.valueOf(0.0) : threshold;
    }

}
