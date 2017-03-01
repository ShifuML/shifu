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
package ml.shifu.shifu.core;

/**
 * Class for training convergence judging.
 * 
 * @author xiaobzheng (zheng.xiaobin.roubao@gmail.com)
 */

public class ConvergeJudger {

    /**
     * Compare threshold and error, if error &lt;= threshold then return true, else false.
     * 
     * @param error
     *            the error
     * @param threshold
     *            the threshold
     * @return threshold and error compare result.
     */
    public boolean judge(double error, double threshold) {
        return Double.compare(error, threshold) <= 0;
    }

}
