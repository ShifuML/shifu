/*
 * Copyright [2012-2014] PayPal Software Foundation
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
package ml.shifu.shifu.actor.worker;

import java.util.ArrayList;
import java.util.List;

import ml.shifu.shifu.container.ValueObject;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ColumnType;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.BasicStatsCalculator;
import ml.shifu.shifu.core.Binning;
import ml.shifu.shifu.core.Binning.BinningDataType;
import ml.shifu.shifu.core.ColumnStatsCalculator;
import ml.shifu.shifu.core.ColumnStatsCalculator.ColumnMetrics;
import ml.shifu.shifu.message.StatsResultMessage;
import ml.shifu.shifu.message.StatsValueObjectMessage;

import org.apache.commons.collections.CollectionUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import akka.actor.ActorRef;

/**
 * StatsCalculateWorker class calculates the stats for each column
 * It will do the binning for the column, calculate max/min/average, and calculate KS/IV
 */
public class StatsCalculateWorker extends AbstractWorkerActor {

    private static Logger log = LoggerFactory.getLogger(StatsCalculateWorker.class);
    private List<ValueObject> voList;
    private int receivedMsgCnt;
    private long missing;
    private long total;

    public StatsCalculateWorker(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, ActorRef parentActorRef,
            ActorRef nextActorRef) {
        super(modelConfig, columnConfigList, parentActorRef, nextActorRef);
        voList = new ArrayList<ValueObject>();
        receivedMsgCnt = 0;
        missing = 0l;
        total = 0;
    }

    @Override
    public void handleMsg(Object message) {
        if(message instanceof StatsValueObjectMessage) {
            StatsValueObjectMessage statsVoMessage = (StatsValueObjectMessage) message;
            voList.addAll(statsVoMessage.getVoList());
            this.missing += statsVoMessage.getMissing();
            this.total += statsVoMessage.getTotal();
            receivedMsgCnt++;

            if(receivedMsgCnt == statsVoMessage.getTotalMsgCnt()) {
                ColumnConfig columnConfig = columnConfigList.get(statsVoMessage.getColumnNum());
                calculateColumnStats(columnConfig, voList);
                columnConfig.setMissingCnt(this.missing);
                columnConfig.setTotalCount(this.total);
                columnConfig.setMissingPercentage((double) missing / total);
                parentActorRef.tell(new StatsResultMessage(columnConfig), this.getSelf());
            }
        } else {
            unhandled(message);
        }

    }

    /**
     * Do the stats calculation
     * 
     * @param columnConfig
     * @param valueObjList
     */
    private void calculateColumnStats(ColumnConfig columnConfig, List<ValueObject> valueObjList) {
        if(CollectionUtils.isEmpty(valueObjList)) {
            log.error("No values for column : {}, please check!", columnConfig.getColumnName());
            return;
        }

        BinningDataType dataType;
        if(columnConfig.isNumerical()) {
            dataType = BinningDataType.Numerical;
        } else if(columnConfig.isCategorical()) {
            dataType = BinningDataType.Categorical;
        } else {
            dataType = BinningDataType.Auto;
        }

        // Binning
        Binning binning = new Binning(modelConfig.getPosTags(), modelConfig.getNegTags(), dataType, valueObjList);
        log.info("posTags - {}, negTags - {}, first example tag - {}", modelConfig.getPosTags(), modelConfig.getNegTags(), valueObjList.get(0).getTag());
        binning.setMaxNumOfBins(modelConfig.getBinningExpectedNum());
        binning.setBinningMethod(modelConfig.getBinningMethod());
        binning.setAutoTypeThreshold(modelConfig.getAutoTypeThreshold());
        binning.setMergeEnabled(Boolean.TRUE);
        binning.doBinning();

        // Calculate Basic Stats
        BasicStatsCalculator basicStatsCalculator = new BasicStatsCalculator(binning.getUpdatedVoList(),
                modelConfig.getNumericalValueThreshold());

        // Calculate KSIV, based on Binning result
        ColumnMetrics columnCountMetrics = ColumnStatsCalculator.calculateColumnMetrics(binning.getBinCountNeg(),
                binning.getBinCountPos());
        ColumnMetrics columnWeightMetrics = ColumnStatsCalculator.calculateColumnMetrics(binning.getBinWeightedNeg(),
                binning.getBinWeightedPos());

        dataType = binning.getUpdatedDataType();
        if(dataType.equals(BinningDataType.Numerical)) {
            columnConfig.setColumnType(ColumnType.N);
            columnConfig.setBinBoundary(binning.getBinBoundary());
        } else {
            columnConfig.setColumnType(ColumnType.C);
            columnConfig.setBinCategory(binning.getBinCategory());
        }

        columnConfig.setBinCountNeg(binning.getBinCountNeg());
        columnConfig.setBinCountPos(binning.getBinCountPos());
        columnConfig.setBinPosCaseRate(binning.getBinPosCaseRate());
        columnConfig.setKs(columnCountMetrics.getKs());
        columnConfig.setIv(columnCountMetrics.getIv());
        columnConfig.getColumnStats().setWoe(columnCountMetrics.getWoe());
        columnConfig.getColumnStats().setWeightedKs(columnWeightMetrics.getKs());
        columnConfig.getColumnStats().setWeightedIv(columnWeightMetrics.getIv());
        columnConfig.getColumnStats().setWeightedWoe(columnWeightMetrics.getWoe());

        columnConfig.setMax(basicStatsCalculator.getMax());
        columnConfig.setMin(basicStatsCalculator.getMin());
        columnConfig.setMean(basicStatsCalculator.getMean());
        columnConfig.setStdDev(basicStatsCalculator.getStdDev());
        columnConfig.setMedian(basicStatsCalculator.getMedian());
        columnConfig.setBinWeightedNeg(binning.getBinWeightedNeg());
        columnConfig.setBinWeightedPos(binning.getBinWeightedPos());
        columnConfig.getColumnBinning().setBinCountWoe(columnCountMetrics.getBinningWoe());
        columnConfig.getColumnBinning().setBinWeightedWoe(columnWeightMetrics.getBinningWoe());

        // columnConfig.setMissingCnt(cnt)
    }
}
