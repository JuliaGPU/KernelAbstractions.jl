import reframe as rfm
import reframe.utility.sanity as sn

# Requires a path
class JuliaTest(rfm.RegressionTest):
    executable = 'julia'
    executable_opts = ['--project=.']
    tags = {'julia'}
    build_system = 'CustomBuild'

    julia_script = ''
    julia_script_opts = []

    @run_before('compile')
    def setup_build(self):
        self.build_system.commands = [
            'julia --project=. -e "import Pkg; Pkg.resolve()"',
            'julia --project=. -e "import Pkg; Pkg.instantiate()"',
        ]

    @run_before('run')
    def set_executable_opts(self):
        self.executable_opts.append(self.julia_script)
        self.executable_opts.extend(self.julia_script_opts)


@rfm.simple_test
class saxpy_test(JuliaTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']

    julia_script = 'saxpy.jl'

    @sanity_function
    def validate(self):
        return sn.assert_found(r'Solution Validates', self.stdout)

    @performance_function('MB/s')
    def copy_bw(self):
        return sn.extractsingle(r'Copy:\s+(\S+)', self.stdout, 1, float)

    @performance_function('MB/s')
    def saxpy_bw(self):
        return sn.extractsingle(r'Saxpy:\s+(\S+)', self.stdout, 1, float)
