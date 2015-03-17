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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author xiaobzheng
 * 
 */
public class ConvergeJudger {

    private static final Logger log = LoggerFactory.getLogger(ConvergeJudger.class);
            
    public static boolean isConverged(double train_err, double test_err, double threshold) {
        double avgErr = (train_err + test_err) / 2;
        log.info("Average err now is: " + avgErr);
        return avgErr <= threshold ? true : false;
    }

}

