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
import ml.shifu.shifu.di.spi.Normalizer;

public class NormalizationModule extends AbstractModule {

    private Class normalizerImplClass;

    public NormalizationModule(String className) {
        try {
            normalizerImplClass = Class.forName(className);
        } catch (Exception e) {
            throw new RuntimeException("No such implementation class: " + className);
        }
    }

    @Override
    protected void configure() {
        bind(Normalizer.class).to(normalizerImplClass);
    }
}
