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

package ml.shifu.plugin.spark.trainer;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningSchema;

import com.google.common.primitives.Ints;

public class SparkCommonUtil {

	public static List<Integer> getFieldIDViaUsageType(MiningSchema schema,
			FieldUsageType usageType) {
		List<MiningField> miningFields = schema.getMiningFields();
		List<Integer> idList = new ArrayList<Integer>();
		for (int i = 0; i < miningFields.size(); i++) {
			if (miningFields.get(i).getUsageType().equals(usageType))
				idList.add(i);
		}
		return idList;
	}

	public static int[] getActiveFields(MiningSchema schema) {
		int[] activeFields = Ints.toArray(getFieldIDViaUsageType(schema,
				FieldUsageType.ACTIVE));
		return activeFields;
	}

}
