/*
 * Copyright [2013-2017] PayPal Software Foundation
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
package ml.shifu.shifu.udf;

import java.io.IOException;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.Tuple;

/**
 * TODO
 */
public class ColumnProjector extends AbstractTrainerUDF<Tuple> {

    private EvalConfig evalConfig;

    @SuppressWarnings("unused")
    private String scoreMetaColumn;

    private String[] headers;
    
    @SuppressWarnings("unused")
    private int targetColumnIndex = -1;
    
    @SuppressWarnings("unused")
    private int weightColumnIndex = -1;
    
    @SuppressWarnings("unused")
    private int scoreMetaColumnIndex = -1;

    public ColumnProjector(String source, String pathModelConfig, String pathColumnConfig) throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
    }

    public ColumnProjector(String source, String pathModelConfig, String pathColumnConfig, String evalSetName,
            String columnName) throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
        this.evalConfig = modelConfig.getEvalConfigByName(evalSetName);
        this.scoreMetaColumn = columnName;
        
        // create model runner
        if(StringUtils.isNotBlank(evalConfig.getDataSet().getHeaderPath())) {
            this.headers = CommonUtils.getHeaders(evalConfig.getDataSet().getHeaderPath(), evalConfig.getDataSet()
                    .getHeaderDelimiter(), evalConfig.getDataSet().getSource());
        } else {
            String delimiter = StringUtils.isBlank(evalConfig.getDataSet().getHeaderDelimiter()) ? evalConfig
                    .getDataSet().getDataDelimiter() : evalConfig.getDataSet().getHeaderDelimiter();
            String[] fields = CommonUtils.takeFirstLine(evalConfig.getDataSet().getDataPath(), delimiter, evalConfig
                    .getDataSet().getSource());
            if(StringUtils.join(fields, "").contains(modelConfig.getTargetColumnName())) {
                this.headers = new String[fields.length];
                for(int i = 0; i < fields.length; i++) {
                    this.headers[i] = CommonUtils.getRelativePigHeaderColumnName(fields[i]);
                }
                log.warn("No header path is provided, we will try to read first line and detect schema.");
                log.warn("Schema in ColumnConfig.json are named as first line of data set path.");
            } else {
                log.warn("No header path is provided, we will try to read first line and detect schema.");
                log.warn("Schema in ColumnConfig.json are named as  index 0, 1, 2, 3 ...");
                log.warn("Please make sure weight column and tag column are also taking index as name.");
                this.headers = new String[fields.length];
                for(int i = 0; i < fields.length; i++) {
                    this.headers[i] = i + "";
                }
            }
        }
        
        for(int i = 0; i < this.headers.length; i++) {
//            if()
            
        }
    }

    @Override
    public Tuple exec(Tuple input) throws IOException {
        //
        return null;
    }

}
