"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from providence.type_utils import once


class TestOnce:
    def test__invoked_only_once(self):
        @once
        def fun(x):
            return x * 2

        out = fun(5)
        assert out == 10
        none = fun(5)
        assert none is None

    def test__invoked_only_once__counter(self):
        class C:
            @once
            def __call__(self, x) -> int:
                return x * 2

        inst = C()

        out = inst(5)
        assert out == 10
        assert inst(5) is None

    def test__invoked_only_once__counter_alternate(self):
        class C:
            count = 0

            @once
            def foo(self, x) -> int:
                self.count += 1
                return x * 2

        inst = C()

        out = inst.foo(5)
        assert out == 10
        assert inst.count == 1
        assert inst.foo(5) is None
        assert inst.count == 1

    def test__invoked_only_once__class_counter(self):
        class C:
            count = 0

            @once
            @classmethod
            def class_call(cls, x) -> int:
                cls.count += 1
                return x * 2

        out = C.class_call(5)
        assert out == 10
        assert C.count == 1
        assert C.class_call(5) is None
        assert C.count == 1
