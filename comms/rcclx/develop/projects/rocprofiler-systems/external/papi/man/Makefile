clean:
	rm -f *~ core man3/*~

install:
	@echo "Man pages (MANDIR) being installed in: \"$(MANDIR)\""; 
	-mkdir -p $(MANDIR)/man3
	-chmod go+rx $(MANDIR)/man3
	-cp man3/PAPI*.3 $(MANDIR)/man3
	-chmod go+r $(MANDIR)/man3/PAPI*.3
	-mkdir -p $(MANDIR)/man1
	-chmod go+rx $(MANDIR)/man1
	-cp man1/*.1 $(MANDIR)/man1
	-chmod go+r $(MANDIR)/man1/*.1
