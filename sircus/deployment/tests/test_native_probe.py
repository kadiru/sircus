import copy
import json
from pathlib import Path
import unittest
from sircus.deployment.check_native import validate_config, validate_inventory


class NativeProbeGuards(unittest.TestCase):
    def config(self):
        value = json.loads((Path(__file__).parents[1]/"viam-fragment.template.json").read_text())
        value['network'] = {'bind_address': '127.0.0.1:18768'}
        value['modules'][0]['executable_path'] = str(Path(__file__).parents[1]/'run-module.sh')
        return value

    def test_local_only_and_no_hardware_cloud_or_remotes(self):
        config = self.config()
        validate_config(config, '127.0.0.1:18768')
        for target in ('sfhack10.viam.cloud:443','192.168.1.2:18768','localhost:18768','127.0.0.1:0'):
            with self.assertRaises(ValueError):validate_config(config,target)
        for field in ('components','remotes','cloud','auth'):
            changed=copy.deepcopy(config);changed[field]=[]
            with self.assertRaises(ValueError):validate_config(changed,'127.0.0.1:18768')

    def test_rejects_dependencies_and_unexpected_environment(self):
        for mutation in ('dependencies','environment','mode'):
            config=self.config()
            if mutation=='dependencies':config['services'][0]['depends_on']=['arm']
            if mutation=='environment':config['modules'][0]['env']={'VIAM_API_KEY':'TEST_ONLY'}
            if mutation=='mode':config['services'][0]['attributes']['mode']='hardware'
            with self.assertRaises(ValueError):validate_config(config,'127.0.0.1:18768')

    def test_resource_inventory_exact_and_remote_prefix_rejected(self):
        resources=[{'namespace':'rdk','type':'service','subtype':kind,'name':name}
                   for kind,name in [('generic','sircus-operation-coordinator'),('vision','sircus-bucket-vision')]]
        validate_inventory(resources)
        for extra in ({'namespace':'rdk','type':'component','subtype':'arm','name':'arm'},
                      {'namespace':'rdk','type':'service','subtype':'generic','name':'remote:sircus-coordinator-rehearsal'}):
            with self.assertRaises(ValueError):validate_inventory(resources+[extra])


if __name__=='__main__':unittest.main()
