package ml.shifu.shifu.core.processor;

import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData;
import ml.shifu.shifu.core.DataPurifier;
import ml.shifu.shifu.core.validator.ModelInspector;
import ml.shifu.shifu.util.HdfsPartFile;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Map;

/**
 * Copyright [2013-2018] PayPal Software Foundation
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License")
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **/

public class ShifuTestProcessor extends BasicModelProcessor {

    public static final Logger LOG = LoggerFactory.getLogger(ShifuTestProcessor.class);

    public static final String TEST_RECORD_CNT = "TEST_RECORD_CNT";
    public static final String IS_TO_TEST_FILTER = "IS_TO_TEST_FILTER";
    public static final String TEST_TARGET = "TEST_TARGET";

    public ShifuTestProcessor(Map<String, Object> params) {
        this.params = params;
    }

    public int run() {
        LOG.info("Step Start: test");
        int status = 0;

        try {
            setUp(ModelInspector.ModelStep.TEST);

            if(isToTestFilter()) {
                String testTarget = StringUtils.trimToEmpty(getTestTarget());
                if(StringUtils.isBlank(testTarget)) { // test filter in training dataset
                    status = runFilterTest(modelConfig);
                } else if(testTarget.equals("*")) { // test filters in train and eval dataset
                    status = runFilterTest(modelConfig);
                    for (EvalConfig evalConfig : this.modelConfig.getEvals()) {
                        status +=  runFilterTest(evalConfig);
                    }
                } else {
                    String[] evalNames = testTarget.split(",");
                    for (String evalName : evalNames) {
                        EvalConfig evalConfig = this.modelConfig.getEvalConfigByName(StringUtils.trimToEmpty(evalName));
                        if ( evalConfig == null ) {
                            LOG.error("Eval - {} doesn't exist!");
                            status = 1;
                            break;
                        }

                        status += runFilterTest(evalConfig);
                    }
                }
            }
        } catch (Exception e) {
            LOG.error("Fail to run test for Shifu.", e);
            status = 1;
        }

        return (status > 0 ? 1 : 0);
    }

    private int runFilterTest(ModelConfig modelConfig) throws IOException {
        RawSourceData dataset = modelConfig.getDataSet();

        if(StringUtils.isBlank(dataset.getFilterExpressions())) {
            LOG.warn("No filter expression set in train dataset. Skip it!");
            return 0;
        }

        DataPurifier dataPurifier = new DataPurifier(modelConfig);
        return doFilterTest(dataPurifier, dataset);
    }

    private int runFilterTest(EvalConfig evalConfig) throws IOException {
        RawSourceData dataset = evalConfig.getDataSet();
        if(StringUtils.isBlank(dataset.getFilterExpressions())) {
            LOG.warn("No filter expression set in eval-{} dataset. Skip it!", evalConfig.getName());
            return 0;
        }

        DataPurifier dataPurifier = new DataPurifier(evalConfig);
        return doFilterTest(dataPurifier, dataset);
    }

    private int doFilterTest(DataPurifier dataPurifier, RawSourceData dataset) throws IOException {
        HdfsPartFile hdfsPartFile = new HdfsPartFile(dataset.getDataPath(), dataset.getSource());
        int totalLineCnt = 0;
        int matchLineCnt = 0;
        int testRecordCnt = getTestRecordCnt();
        while (totalLineCnt < testRecordCnt) {
            String record = hdfsPartFile.readLine();

            if ( record == null ) {
                break;
            }

            boolean isMatched = dataPurifier.isFilter(record);
            if ( isMatched ) {
                matchLineCnt ++;
            }
            totalLineCnt ++;
        }

        LOG.info("Filter Result:");
        LOG.info("\t {} out of {} records are matched by the filter expression.", matchLineCnt, totalLineCnt);
        return 0;
    }

    private boolean isToTestFilter() {
        return getBooleanParam(this.params, IS_TO_TEST_FILTER);
    }

    private String getTestTarget() {
        return getStringParam(this.params, TEST_TARGET);
    }

    private int getTestRecordCnt() {
        return getIntParam(this.params, TEST_RECORD_CNT, 100);
    }
}
