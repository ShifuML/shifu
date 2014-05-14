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

package ml.shifu.shifu.di.service;


import com.google.inject.Inject;
import ml.shifu.shifu.container.obj.ColumnRawStatsResult;
import ml.shifu.shifu.di.spi.ColumnRawStatsCalculator;
import ml.shifu.shifu.container.RawValueObject;

import java.util.List;

public class ColumnRawStatsService {

    private ColumnRawStatsCalculator screener;

    @Inject
    public ColumnRawStatsService(ColumnRawStatsCalculator screener) {
        this.screener = screener;
    }

    public ColumnRawStatsResult getResult(List<RawValueObject> rvoList, List<String> posTags, List<String> negTags) {
        return screener.calculate(rvoList, posTags, negTags);
    }

}
