package ml.shifu.shifu.core.dvarsel.wrapper;
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.dvarsel.AbstractWorkerConductor;
import ml.shifu.shifu.core.dvarsel.VarSelMasterResult;
import ml.shifu.shifu.core.dvarsel.VarSelWorkerResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created on 11/24/2014.
 */
public class WrapperWorkerConductor extends AbstractWorkerConductor {

    private static final Logger LOG = LoggerFactory.getLogger(WrapperWorkerConductor.class);

    private List<ColumnConfig> candidates;
    private Set<Integer> baseColumnSet;

    public WrapperWorkerConductor(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        super(modelConfig, columnConfigList);

        this.candidates = new ArrayList<ColumnConfig>();
        for ( ColumnConfig columnConfig : columnConfigList ) {
            if ( columnConfig.isCandidate() && !columnConfig.isForceSelect() ) {
                candidates.add(columnConfig);
            }
        }
    }

    @Override
    public void consumeMasterResult(VarSelMasterResult masterResult) {
        baseColumnSet = new HashSet<Integer>(masterResult.getColumnIdList());
    }

    @Override
    public VarSelWorkerResult generateVarSelResult() {
        Set<Integer> workingColumnSet = new HashSet<Integer>();

        double minValidateError = Double.MAX_VALUE;
        ColumnConfig bestCandidate = null;

        for ( ColumnConfig columnConfig : candidates ) {
            if ( !baseColumnSet.contains(columnConfig.getColumnNum()) ) {
                workingColumnSet.clear();
                workingColumnSet.addAll(baseColumnSet);
                workingColumnSet.add(columnConfig.getColumnNum());

                ValidationConductor validationConductor =
                        new ValidationConductor(modelConfig, columnConfigList, workingColumnSet, trainingDataSet);
                double validateError = validationConductor.runValidate();
                if ( validateError < minValidateError ) {
                    minValidateError = validateError;
                    bestCandidate = columnConfig;
                }
            }
        }

        LOG.info("find best variable - " + bestCandidate.getColumnName() + ", with error - " + minValidateError);

        return ((bestCandidate == null) ? getDefaultWorkerResult() : getWorkerResult(bestCandidate.getColumnNum()));
    }

    @Override
    public VarSelWorkerResult getDefaultWorkerResult() {
        return getWorkerResult(-1);
    }

    private VarSelWorkerResult getWorkerResult(int columnId) {
        List<Integer> columnIdList = new ArrayList<Integer>();
        columnIdList.add(columnId);
        return new VarSelWorkerResult(columnIdList);
    }
}
