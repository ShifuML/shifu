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
package ml.shifu.shifu.pig;

import java.io.IOException;
import java.util.Map;
import java.util.Properties;

import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Environment;

import org.apache.pig.ExecType;
import org.apache.pig.PigServer;
import org.apache.pig.backend.executionengine.ExecException;
import org.apache.pig.impl.PigContext;
import org.apache.pig.impl.logicalLayer.FrontendException;
import org.apache.pig.newplan.logical.relational.LogicalPlan;
import org.apache.pig.tools.pigstats.PigStats;

/**
 * {@link ShifuPigServer} is add an entry point to inject properties set in shifuconfig or command to make sure user
 * configured property can override properties in 'SET' of pig script.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class ShifuPigServer extends PigServer {

    public ShifuPigServer(ExecType execType) throws ExecException {
        super(execType);
    }

    public ShifuPigServer(ExecType execType, Properties properties) throws ExecException {
        super(execType, properties);
    }

    public ShifuPigServer(PigContext context, boolean connect) throws ExecException {
        super(context, connect);
    }

    public ShifuPigServer(PigContext context) throws ExecException {
        super(context);
    }

    public ShifuPigServer(String execTypeString) throws ExecException, IOException {
        super(execTypeString);
    }

    protected PigStats launchPlan(LogicalPlan lp, String jobName) throws ExecException, FrontendException {
        // add logic to update properties and make config in Environment.getProperties() override properties in pig
        // script
        for(Map.Entry<Object, Object> entry: Environment.getProperties().entrySet()) {
            if(CommonUtils.isHadoopConfigurationInjected(entry.getKey().toString())) {
                super.pigContext.getProperties().put(entry.getKey(), entry.getValue());
            }
        }

        return super.launchPlan(lp, jobName);
    }
}
