.PHONY: bundle verify release

bundle:
	./bundle.sh

verify:
	@:[ -n "$$TAG" ] || (echo "usage: make verify TAG=release-YYYYMMDD-p39" && exit 1)
	./tools/verify_release.sh "$$TAG"

release:
	@:[ -n "$$TAG" ] || (echo "usage: make release TAG=release-YYYYMMDD-p39" && exit 1)
	@DATE=$$(echo "$$TAG" | sed -E 's/^release-([0-9]{8})-p39$$/\1/') && \
	  gh release create "$$TAG" "p39_release_$${DATE}.tgz" "p39_release_checksums.txt" \
	    -t "p39 Release" -F release_bundle/RELEASE_NOTES.md
