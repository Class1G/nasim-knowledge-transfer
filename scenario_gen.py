import random

from nasim.scenarios.scenario import Scenario
from nasim.scenarios.host import Host

OS =        ['windows', 'linux']
SERVICES =  ['http', 'ftp', 'ssh']
PROCESSES = ['tomcat', 'daclsvc']

# HOST configurations [OS, SERVICE, PROCESS]
HOST_DUMMY = [['linux',     'none', 'none'],
              ['linux',     'ftp',  'none'],
              ['windows',   'none', 'none'],
              ['windows',   'ftp',  'none'],
              ['windows',   'ssh',  'none']]

HOST_EXPLOITABLE = [['linux',   'http', 'none'],
                    ['linux',   'http', 'daclsvc'],
                    ['linux',   'ssh',  'none'],
                    ['linux',   'ssh',  'daclsvc'],
                    ['windows', 'http', 'none'],
                    ['windows', 'http', 'tomcat']]

HOST_ROOTABLE = [['linux',  'http', 'tomcat'],
                 ['linux',  'ssh',  'tomcat'],
                 ['windows','http', 'daclsvc']]

TOPOLOGY = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
STEPLIMIT = 1000

# ---------------------

# 0 - non. expoitable
# 1 - only expoitable
# 2 - rootable
HOST_CONFIGS = [[0, 0, 1, 2], [0, 1, 0, 2], [0, 1, 2, 0], [1, 0, 0, 2], [1, 0, 2, 0], [1, 2, 0, 0], 
                [0, 0, 2, 1], [0, 2, 0, 1], [0, 2, 1, 0], [2, 0, 0, 1], [2, 0, 1, 0], [2, 1, 0, 0]]
                  
SUBNET_CONFIGS = [([1, 1, 3], [(1, 0), (2, 0), (2, 1), (2, 2)]), 
                 ([1, 2, 2], [(1, 0), (1, 1), (2, 0), (2, 1)]), 
                 ([1, 3, 1], [(1, 0), (1, 1), (1, 2), (2, 0)])]


def generate_small_scenarios (seed = None, 
                              os_scan_cost =        1,
                              service_scan_cost =   1,
                              subnet_scan_cost =    1,
                              process_scan_cost =   1,
                              exploit_cost =        3,
                              privesc_cost =        1,
                              discovery_value =     0,
                              sensitive_value =     100,
                                ):
    
    random.seed(seed)
    scenario_dict = dict()

    # Choose configurations
    subnet_config = random.choice(SUBNET_CONFIGS)
    host_config =   random.choice(HOST_CONFIGS)
    
    # Construct predefined structures
    # Exploits
    exploits = dict()
    exploits['e_http'] = {'os': None,
                          'service': 'http',
                          'access': 1,
                          'cost': exploit_cost,
                          'prob': 1}
    
    exploits['e_ssh'] = {'os': 'linux',
                         'service': 'ssh',
                         'access': 1,
                         'cost': exploit_cost,
                         'prob': 1}
    
    
    # PrivEscs
    privescs = dict()
    privescs['pe_tomcat'] = {'os': 'linux',
                             'process': 'tomcat',
                             'access': 2,
                             'cost': privesc_cost,
                             'prob': 1}
    
    privescs['pe_daclsvc'] = {'os': 'windows',
                             'process': 'daclsvc',
                             'access': 2,
                             'cost': privesc_cost,
                             'prob': 1}
    
    
    # Construct variable structures
    # Sensitive hosts
    s_hosts = dict()
    index = host_config.index(2)
    s_adr = subnet_config[1][index]
    s_hosts[s_adr] = sensitive_value
    
    # Firewall
    firewall = {(0, 1): SERVICES[:],
                (1, 0): SERVICES[:],
                (0, 2): SERVICES[:],
                (2, 0): SERVICES[:],
                (1, 2): SERVICES[:], 
                (2, 1): SERVICES[:]}
    
    # Hosts
    hosts = dict()
    
    for i in range(4):
        v = 0
        # determine type of host (dummy, exploitable, rootable)
        host_type = host_config[i]
        if host_type == 0: # Dummy
            host_c = random.choice(HOST_DUMMY)
        elif host_type == 1: # Exploitable
            host_c = random.choice(HOST_EXPLOITABLE)
        elif host_type == 2: # Rootable
            host_c = random.choice(HOST_ROOTABLE)
            v = sensitive_value
        else:
            raise ValueError('Unexpected host_config sample')
        

        os_c = {}
        for os in OS:
            os_c[os] = (host_c[0] == os) 
            
        service_c = {}
        for service in SERVICES:
            service_c[service] = (host_c[1] == service)
            
        process_c = {}
        for process in PROCESSES:
            process_c[process] = (host_c[2] == process)
        
        
        adr = subnet_config[1][i]
        hosts[adr] = Host(address = adr,
                          os = os_c,
                          services = service_c,
                          processes = process_c,
                          firewall = {},
                          value = v)

    
    # Construct final object
    scenario_dict['subnets'] = subnet_config[0]
    scenario_dict['address_space_bounds'] = (3, 3)
    scenario_dict['topology'] = TOPOLOGY
    scenario_dict['os'] = OS
    scenario_dict['services'] = SERVICES
    scenario_dict['processes'] = PROCESSES
    scenario_dict['sensitive_hosts'] = s_hosts
    scenario_dict['exploits'] = exploits
    scenario_dict['privilege_escalation'] = privescs
    scenario_dict['os_scan_cost'] = os_scan_cost
    scenario_dict['service_scan_cost'] = service_scan_cost
    scenario_dict['subnet_scan_cost'] = subnet_scan_cost
    scenario_dict['process_scan_cost'] = process_scan_cost
    scenario_dict['firewall'] = firewall
    scenario_dict['host'] = hosts
    scenario_dict['step_limit'] = STEPLIMIT
    
    return Scenario(scenario_dict, name = None, generated = False)

def generate_part_scenarios (seed = None, 
                              os_scan_cost = 1,
                              service_scan_cost = 1,
                              subnet_scan_cost = 1,
                              process_scan_cost = 1,
                              exploit_cost = 3,
                              privesc_cost = 1,
                              discovery_value = 0,
                              sensitive_value = 100,
                                ):
    
    random.seed(seed)
    scenario_dict = dict()

    # Choose configurations
    subnet_config = random.choice(SUBNET_CONFIGS)
    host_config =   random.choice(HOST_CONFIGS)
    
    
    
    # Partially observable check and application
    topology = TOPOLOGY    
    if subnet_config[0] == [1, 1, 3]:
        topology = [[1, 0, 1], [0, 1, 1], [1, 1, 1]]
    elif subnet_config[0] == [1, 3, 1]:
        topology = [[1, 1, 0], [1, 1, 1], [0, 1, 1]]
    
    
    
    
    # Construct predefined structures
    # Exploits
    exploits = dict()
    exploits['e_http'] = {'os':         None,
                          'service':    'http',
                          'access':     1,
                          'cost':       exploit_cost,
                          'prob':       1}
    
    exploits['e_ssh'] = {'os':      'linux',
                         'service': 'ssh',
                         'access':  1,
                         'cost':    exploit_cost,
                         'prob':    1}
    
    # PrivEscs
    privescs = dict()
    privescs['pe_tomcat'] = {'os':      'linux',
                             'process': 'tomcat',
                             'access':  2,
                             'cost':    privesc_cost,
                             'prob':    1}
    
    privescs['pe_daclsvc'] = {'os':     'windows',
                             'process': 'daclsvc',
                             'access':  2,
                             'cost':    privesc_cost,
                             'prob':    1}
    
    
    # Construct variable structures
    # Sensitive hosts
    s_hosts = dict()
    index = host_config.index(2)
    s_adr = subnet_config[1][index]
    s_hosts[s_adr] = sensitive_value
    
    # Firewall
    firewall = {(0, 1): SERVICES[:],
                (1, 0): SERVICES[:],
                (0, 2): SERVICES[:],
                (2, 0): SERVICES[:],
                (1, 2): SERVICES[:], 
                (2, 1): SERVICES[:]}
    
    # Hosts
    hosts = dict()
    
    for i in range(4):
        v = 0
        # determine type of host (dummy, exploitable, rootable)
        host_type = host_config[i]
        if host_type == 0: # Dummy
            host_c = random.choice(HOST_DUMMY)
        elif host_type == 1: # Exploitable
            host_c = random.choice(HOST_EXPLOITABLE)
        elif host_type == 2: # Rootable
            host_c = random.choice(HOST_ROOTABLE)
            v = sensitive_value
        else:
            raise ValueError('Unexpected host_config sample')
        
        os_c = {}
        for os in OS:
            os_c[os] = (host_c[0] == os) 
            
        service_c = {}
        for service in SERVICES:
            service_c[service] = (host_c[1] == service)
            
        process_c = {}
        for process in PROCESSES:
            process_c[process] = (host_c[2] == process)
        
        
        adr = subnet_config[1][i]
        hosts[adr] = Host(address =     adr,
                          os =          os_c,
                          services =    service_c,
                          processes =   process_c,
                          firewall =    {},
                          value =       v)

    # Construct final object
    scenario_dict['subnets'] = subnet_config[0]
    scenario_dict['address_space_bounds'] = (3, 3)
    scenario_dict['topology'] = topology
    scenario_dict['os'] = OS
    scenario_dict['services'] = SERVICES
    scenario_dict['processes'] = PROCESSES
    scenario_dict['sensitive_hosts'] = s_hosts
    scenario_dict['exploits'] = exploits
    scenario_dict['privilege_escalation'] = privescs
    scenario_dict['os_scan_cost'] = os_scan_cost
    scenario_dict['service_scan_cost'] = service_scan_cost
    scenario_dict['subnet_scan_cost'] = subnet_scan_cost
    scenario_dict['process_scan_cost'] = process_scan_cost
    scenario_dict['firewall'] = firewall
    scenario_dict['host'] = hosts
    scenario_dict['step_limit'] = STEPLIMIT
    
    return Scenario(scenario_dict, name = None, generated = False)