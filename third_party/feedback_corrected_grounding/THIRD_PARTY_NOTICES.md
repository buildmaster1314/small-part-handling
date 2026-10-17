# Third-Party Notices

## [LeWM](https://github.com/lucas-maes/le-wm)

`third_party/lewm/` contains an unmodified, complete snapshot of the tracked
upstream contents at commit
`8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`, excluding upstream Git metadata. The
snapshot includes the upstream README and license, training and evaluation entry
points, Hydra configurations, assets, and the `module.py`, `jepa.py`, and
`utils.py` implementation modules. See `third_party/lewm/UPSTREAM.md` for
provenance details. FCPG-specific adaptations live outside the vendored snapshot;
`fcpg/regularization.py` adapts the `SIGReg` implementation from LeWM's
`module.py`.

MIT License

Copyright (c) 2026 Lucas Maes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
