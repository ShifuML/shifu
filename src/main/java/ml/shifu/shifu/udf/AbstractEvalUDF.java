package ml.shifu.shifu.udf;

import ml.shifu.shifu.container.obj.EvalConfig;
import org.apache.commons.collections.CollectionUtils;

import java.io.IOException;
import java.util.HashSet;
import java.util.List;

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

public abstract class AbstractEvalUDF<T> extends AbstractTrainerUDF<T> {

    protected EvalConfig evalConfig;

    public AbstractEvalUDF(String source, String pathModelConfig, String pathColumnConfig,
            String evalSetName) throws IOException {
        super(source, pathModelConfig, pathColumnConfig);

        this.evalConfig = modelConfig.getEvalConfigByName(evalSetName);
        if ( this.evalConfig != null ) {
            this.posTagSet = new HashSet<String>();
            List<String> posTags = this.modelConfig.getPosTags(evalConfig);
            if (CollectionUtils.isNotEmpty(posTags)) {
                this.posTagSet.addAll(posTags);
            }

            this.negTagSet = new HashSet<String>();
            List<String> negTags = this.modelConfig.getNegTags(evalConfig);
            if (CollectionUtils.isNotEmpty(negTags)) {
                this.negTagSet.addAll(negTags);
            }

            this.tagSet = new HashSet<String>(this.modelConfig
                    .getFlattenTags(this.modelConfig.getPosTags(evalConfig), this.modelConfig.getNegTags(evalConfig)));
        }
    }
}
