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

import java.io.IOException;

import ml.shifu.shifu.util.Constants;

import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author xiaobzheng
 * Factory class for generating ConvergeJudger instances.
 */
public class ConvergeJudgerFactory {

    private static Logger LOG = LoggerFactory.getLogger(ConvergeJudgerFactory.class);

    /**
     * @param criteriaStr the criteria string indicates convergence criteria.
     * @return Convergence Judger
     * @throws IOException
     */
    AbstractConvergeJudger getJudger(String criteriaStr) throws IOException {
        //TODO xiaobin: get judger
        // tmp return. edit later.
        LOG.info("Start getting ConvergeJuder instance :");

        String criteria = StringUtils.trimToEmpty(criteriaStr);
        AbstractConvergeJudger judger = null;
        if (Constants.TRAIN_ERR_ONLY.equalsIgnoreCase(criteria)) {
            LOG.info(choiceLogStr(Constants.TRAIN_ERR_ONLY));
            judger = new TrainErrOnlyConvergeJudger();
        } else if (Constants.VALID_ERR_ONLY.equalsIgnoreCase(criteria)) {
            LOG.info(choiceLogStr(Constants.VALID_ERR_ONLY));
            judger = new ValidErrOnlyConvergeJudger();
        } else if (Constants.TRAIN_VALID_AVG_ERR.equalsIgnoreCase(criteria)) {
            LOG.info(choiceLogStr(Constants.TRAIN_VALID_AVG_ERR));
            judger = new TrainValidAvgErrConvergeJudger();
        } else {
            throw new RuntimeException("Training Option for convergence criteria is not Valid : " + criteria);
        }
        return judger;
    }
    
    private String choiceLogStr(String criteriaStr) {
        return "Choose convergence criteria : " + criteriaStr;
    }
}
