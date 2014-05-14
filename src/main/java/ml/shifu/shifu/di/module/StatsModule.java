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

package ml.shifu.shifu.di.module;


import com.google.inject.AbstractModule;
import ml.shifu.shifu.di.spi.*;
import ml.shifu.shifu.util.CommonUtils;


public class StatsModule extends AbstractModule {

    private Class rawStatsCalculatorImplClass;
    private Class numStatsCalculatorImplClass;
    private Class binStatsCalculatorImplClass;
    private Class numBinningCalculatorImplClass;
    private Class catBinningCalculatorImplClass;
    private Class statsProcessorImplClass;

    public StatsModule() {}

    public void setRawStatsCalculatorImplClass(String className) {
        rawStatsCalculatorImplClass = CommonUtils.getClass(className);
    }

    public void setRawStatsCalculatorImplClass(Class clazz) {
        rawStatsCalculatorImplClass = clazz;
    }

    public void setNumBinningCalculatorImplClass(String className) {
        numBinningCalculatorImplClass = CommonUtils.getClass(className);
    }
    public void setNumBinningCalculatorImplClass(Class clazz) {
        numBinningCalculatorImplClass = clazz;
    }

    public void setCatBinningCalculatorImplClass(String className) {
        catBinningCalculatorImplClass = CommonUtils.getClass(className);
    }

    public void setCatBinningCalculatorImplClass(Class clazz) {
        catBinningCalculatorImplClass = clazz;
    }

    public void setNumStatsCalculatorImplClass(String className) {
        numStatsCalculatorImplClass = CommonUtils.getClass(className);
    }

    public void setNumStatsCalculatorImplClass(Class clazz) {
        numStatsCalculatorImplClass = clazz;
    }

    public void setBinStatsCalculatorImplClass(String className) {
        binStatsCalculatorImplClass = CommonUtils.getClass(className);
    }

    public void setBinStatsCalculatorImplClass(Class clazz)  {
        binStatsCalculatorImplClass = clazz;
    }

    public void setStatsProcessorImplClass(String className) {
        statsProcessorImplClass = CommonUtils.getClass(className);
    }

    public void setStatsProcessorImplClass(Class clazz) {
        statsProcessorImplClass = clazz;
    }

    @Override
    protected void configure() {

        if (rawStatsCalculatorImplClass != null) {
            bind(ColumnRawStatsCalculator.class).to(rawStatsCalculatorImplClass);
        }

        if (numBinningCalculatorImplClass != null) {
            bind(ColumnNumBinningCalculator.class).to(numBinningCalculatorImplClass);
        }

        if (catBinningCalculatorImplClass != null) {
            bind(ColumnCatBinningCalculator.class).to(catBinningCalculatorImplClass);
        }

        if (numStatsCalculatorImplClass != null) {
            bind(ColumnNumStatsCalculator.class).to(numStatsCalculatorImplClass);
        }

        if (binStatsCalculatorImplClass != null) {
            bind(ColumnBinStatsCalculator.class).to(binStatsCalculatorImplClass);
        }

        if (statsProcessorImplClass != null) {
            bind(StatsProcessor.class).to(statsProcessorImplClass);
        }
    }
}
