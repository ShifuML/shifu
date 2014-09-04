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


package ml.shifu.plugin.spark.norm;

import java.util.List;

import org.dmg.pmml.DataField;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.PMML;

import ml.shifu.core.di.builtin.transform.DefaultTransformationExecutor;
/**
 * This is the class which contains all the variables which are broadcast to the worker nodes. 
 * An object of this class is created and wrapped into a broadcast variable using JavaSparkContext.broadcast().
 * The resulting broadcast variable is passed to all Normalizers running on the worker nodes. 
 * The Normalizers unpack the BroadcastVariables object and use the contained variables.
 */

public class BroadcastVariables {

    private DefaultTransformationExecutor exec;
    private PMML pmml;
    private List<DataField> dataFields;
    private List<DerivedField> activeFields;
    private List<DerivedField> targetFields;
    private String precision;
    private String delimiter;

    public BroadcastVariables(DefaultTransformationExecutor executor,
            PMML pmml, List<DataField> dataFields,
            List<DerivedField> activeFields, List<DerivedField> targetFields,
            String precision, String delimiter) {
        this.exec = executor;
        this.pmml = pmml;
        this.dataFields = dataFields;
        this.activeFields = activeFields;
        this.targetFields = targetFields;
        this.precision = precision;
        this.delimiter = delimiter;
    }

    public PMML getPmml() {
        return pmml;
    }

    public List<DataField> getDataFields() {
        return dataFields;
    }

    public List<DerivedField> getTargetFields() {
        return targetFields;
    }

    public List<DerivedField> getActiveFields() {
        return activeFields;
    }

    public DefaultTransformationExecutor getExec() {
        return exec;
    }

    public String getPrecision() {
        return this.precision;
    }

    public String getDelimiter() {
        return this.delimiter;
    }

}
