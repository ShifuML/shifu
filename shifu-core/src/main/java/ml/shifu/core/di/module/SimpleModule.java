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

package ml.shifu.core.di.module;


import com.google.inject.AbstractModule;
import ml.shifu.core.request.Binding;
import ml.shifu.core.request.Request;
import ml.shifu.core.util.CommonUtils;

import java.util.HashMap;
import java.util.Map;


public class SimpleModule extends AbstractModule {

    private Map<String, String> bindings = new HashMap<String, String>();

    public SimpleModule() {
    }


    public Map<String, String> getBindings() {
        return bindings;
    }

    public void setBindings(Map<String, String> bindings) {
        this.bindings = bindings;
    }

    public void set(String spi, String impl) {
        bindings.put(spi, impl);
    }

    public void set(Binding binding) {
        if (binding != null) {
            bindings.put(binding.getSpi(), binding.getImpl());
        }
    }

    public void set(Request req) {


        this.set(req.getProcessor());

        for (Binding binding : req.getBindings()) {
            this.set(binding);
        }
    }

    public Boolean has(String spi) {
        return bindings.containsKey(spi) && bindings.get(spi) != null;

    }

    @Override
    protected void configure() {

        for (String spiName : bindings.keySet()) {
            Class spi = CommonUtils.getClass("ml.shifu.core.di.spi." + spiName);
            Class impl = CommonUtils.getClass(bindings.get(spiName));
            bind(spi).to(impl);
        }

    }
}
